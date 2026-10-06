# This file is part of tad-dftd3.
# SPDX-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command line interface: the run
===============================

Reads the structure, builds what the chosen path needs, evaluates the
dispersion energy step by step, as :class:`~tad_dftd3.disp.D3Model` does
in one call, and prints the results.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable
from typing import NamedTuple

import torch
from tad_mctc.cli._main import _torch_threads
from tad_mctc.cli._output import print_native_build, print_system_info
from tad_mctc.cli._timing import Timings
from tad_mctc.io.checks import coldfusion_check
from tad_mctc.io.checks.structure import _coldfusion_uses_neighborlist
from tad_mctc.io.read import read_structure
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor import _native
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.neighbor.list import NeighborList, _build_neighborlists
from tad_mctc.typing import Tensor

from .. import model
from ..cutoff import Cutoff
from ..damping import Damping, DampingParam
from ..disp import D3Model, Pairs, default_damping, dispersion2, dispersion3
from ..reference import Reference, _default_reference
from ..sparse import TripleList
from ._args import DTYPES, build_parser
from ._output import (
    print_energy,
    print_gradient,
    print_neighborlists,
    print_parameters,
)

__all__ = ["main"]


def _cutoff(args: argparse.Namespace) -> Cutoff:
    """The real-space cutoffs the arguments set."""
    return Cutoff(
        cn=args.cutoff_cn,
        disp2=args.cutoff_disp2,
        disp3=args.cutoff_disp3,
        width2=args.width2,
        width3=args.width3,
    )


def _term_cutoffs(cutoff: Cutoff, damping: Damping) -> dict[str, float]:
    """The cutoff of each term the damping evaluates, keyed by its name."""
    terms = {"cn": cutoff.cn, "disp2": cutoff.disp2}
    if damping.three is not None:
        terms["disp3"] = cutoff.disp3
    return terms


def _build_neighborlists_timed(
    structure: Structure, cutoffs: dict[str, float], timings: Timings
) -> dict[str, NeighborList]:
    """Build the list of every term in one shared search, as
    ``dftd3(..., sparse=True)`` does, timing each step of the build."""

    def nlist_stage(label: str) -> contextlib.AbstractContextManager[None]:
        return timings.stage(f"nlist: {label}")

    lists = _build_neighborlists(
        structure,
        tuple(cutoffs.values()),  # pyright: ignore[reportCallIssue]
        stage=nlist_stage,
    )
    return dict(zip(cutoffs, lists))


def _pairs(
    args: argparse.Namespace,
    structure: Structure,
    cutoffs: dict[str, float],
    timings: Timings,
) -> dict[str, Pairs]:
    """How the pairs of every term are enumerated: a neighbour list each,
    the periodic image shifts of a cell at each term's own cutoff (as
    ``dftd3`` builds them), or all pairs of a molecule."""
    if args.neighbor == "sparse":
        return dict(_build_neighborlists_timed(structure, cutoffs, timings))

    if structure.lattice is None:
        return dict.fromkeys(cutoffs)

    assert structure.periodic is not None
    shifts: dict[str, Pairs] = {}
    for term, cutoff in cutoffs.items():
        with timings.stage(f"periodic shifts ({term})"):
            shifts[term] = build_periodic_shifts(
                structure.lattice, structure.periodic, cutoff
            )
    return shifts


class _Stages(NamedTuple):
    """The steps of :meth:`D3Model.__call__`, each a function of what the
    one before computed, compiled with ``fullgraph=True`` on request."""

    cn: Callable[[Structure], Tensor]
    weights: Callable[[Tensor, Tensor], Tensor]
    c6: Callable[[Tensor, Tensor], model.AtomicC6]
    c6_matrix: Callable[[model.AtomicC6], Tensor]
    two: Callable[[Structure, Tensor | model.AtomicC6], Tensor]
    three: Callable[[Structure, Tensor | model.AtomicC6], Tensor]


def _stages(
    args: argparse.Namespace,
    d3: D3Model,
    param: DampingParam,
    damping: Damping,
    pairs: dict[str, Pairs | TripleList],
    ref: Reference,
) -> _Stages:
    """The steps of the energy, over the pairs built beforehand."""
    checkpoint = args.mode == "recompute"

    def cn(structure: Structure) -> Tensor:
        pairs_cn = pairs["cn"]
        assert not isinstance(pairs_cn, TripleList)
        return d3.coordination_number(
            structure, pairs_cn, checkpoint=checkpoint
        )

    def weights(numbers: Tensor, cn: Tensor) -> Tensor:
        return model.weight_references(numbers, cn, ref, d3.weighting_function)

    def c6(numbers: Tensor, weights: Tensor) -> model.AtomicC6:
        return model.factored_c6(numbers, weights, ref)

    def c6_matrix(c6: model.AtomicC6) -> Tensor:
        return c6.dense()

    def two(structure: Structure, c6: Tensor | model.AtomicC6) -> Tensor:
        pairs_disp2 = pairs["disp2"]
        assert not isinstance(pairs_disp2, TripleList)
        return dispersion2(
            structure,
            param,
            c6,
            pairs=pairs_disp2,
            damping=damping.two,
            cutoff=d3.cutoff,
            checkpoint=checkpoint,
        )

    def three(structure: Structure, c6: Tensor | model.AtomicC6) -> Tensor:
        assert damping.three is not None
        return dispersion3(
            structure,
            param,
            c6,
            pairs=pairs["disp3"],
            damping=damping.three,
            cutoff=d3.cutoff,
            max_triples=args.max_triples,
            checkpoint=checkpoint,
        )

    stages = _Stages(cn, weights, c6, c6_matrix, two, three)
    if not args.compile:
        return stages
    return _Stages(*(torch.compile(f, fullgraph=True) for f in stages))


def _evaluate(
    args: argparse.Namespace,
    stages: _Stages,
    structure: Structure,
    three_body: bool,
    timings: Timings,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """The atom-resolved two- and three-body energy (``None`` without a
    three-body damping) and, with ``--grad``, the gradient with respect to
    the positions, each step timed on its own."""
    label = "sparse" if args.neighbor == "sparse" else "dense"

    positions = structure.positions
    if args.grad:
        positions = positions.detach().requires_grad_(True)
        structure = structure.replace(positions=positions)

    with timings.stage(f"coordination number ({label})"):
        cn = stages.cn(structure)
    with timings.stage("reference weights"):
        weights = stages.weights(structure.numbers, cn)
    with timings.stage("atomic C6"):
        c6 = stages.c6(structure.numbers, weights)

    # As in `disp.dispersion`: the dense terms take the C6 matrix, the
    # sparse ones the factors, which take no memory quadratic in `nat`.
    c6_terms: Tensor | model.AtomicC6 = c6
    if label == "dense":
        with timings.stage("C6 matrix"):
            c6_terms = stages.c6_matrix(c6)

    with timings.stage(f"two-body ({label})"):
        energy2 = stages.two(structure, c6_terms)
    energy3 = None
    if three_body:
        with timings.stage(f"three-body ({label})"):
            energy3 = stages.three(structure, c6_terms)

    gradient = None
    if args.grad:
        energy = energy2 if energy3 is None else energy2 + energy3
        with timings.stage("gradient (backward)"):
            (gradient,) = torch.autograd.grad(energy.sum(), positions)

    return energy2, energy3, gradient


def _load_parameters(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> DampingParam:
    """The damping parameters of the functional, with an unknown functional
    or a damping variant it does not have reported as a usage error."""
    try:
        return DampingParam.from_functional(
            args.func,
            damping=args.damping,
            atm=args.atm,
            dtype=DTYPES[args.dtype],
            device=torch.device("cuda") if args.cuda else None,
        )
    except (KeyError, ValueError) as e:
        # A `KeyError` quotes its message, which is a sentence here.
        message = e.args[0] if e.args else str(e)
        parser.error(f"--func {args.func}: {message}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``tad_dftd3`` command line tool."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.nlist_only and args.neighbor == "dense":
        parser.error(
            "--nlist-only builds neighbour lists, so it cannot be "
            "combined with '--neighbor dense'."
        )
    if args.nlist_only and args.grad:
        parser.error("--nlist-only computes no energy to differentiate.")
    if args.compile and args.mode == "recompute":
        parser.error(
            "--mode recompute checkpoints in the backward pass, which "
            "does not trace, so it cannot be combined with --compile."
        )

    if args.cuda and not torch.cuda.is_available():
        raise SystemExit("--cuda given but no CUDA device is available.")

    param = _load_parameters(parser, args)

    with _torch_threads(args.omp):
        _run(args, param)
    return 0


def _run(args: argparse.Namespace, param: DampingParam) -> None:
    """Carry out the run the parsed arguments describe."""
    timings = Timings(args.timing)
    d3 = D3Model(cutoff=_cutoff(args))
    damping = default_damping(param)
    cutoffs = _term_cutoffs(d3.cutoff, damping)

    with timings.stage("read structure"):
        structure = read_structure(args.structure, dtype=DTYPES[args.dtype])

    # A neighbour search on the CPU runs through tad-mctc's native
    # extension, and so does the optional check on the structure just read,
    # unless it is a small molecule compared densely. Loading the extension
    # can mean compiling it, so it is its own step rather than part of
    # whichever search happens to come first.
    checks_with_native = args.coldfusion_check and (
        _coldfusion_uses_neighborlist(structure)
    )
    uses_native = checks_with_native or (
        not args.cuda and args.neighbor == "sparse"
    )
    if uses_native:
        with timings.stage("native extension"):
            _native.is_available()

    if args.coldfusion_check:
        with timings.stage("cold-fusion check"):
            coldfusion_check(structure)
    if args.cuda:
        with timings.stage("move to device"):
            structure = structure.to(torch.device("cuda"))

    if args.nlist_only:
        built = _build_neighborlists_timed(structure, cutoffs, timings)
        print_system_info(args.structure, structure)
        if uses_native:
            print_native_build(_native.build_info())
        print_neighborlists(built)
        timings.report()
        return

    lists = _pairs(args, structure, cutoffs, timings)
    pairs: dict[str, Pairs | TripleList] = dict(lists)
    nbl_disp3 = pairs.get("disp3")
    if args.compile and isinstance(nbl_disp3, NeighborList):
        # The triples of a list are enumerated when the term is called,
        # which does not trace: under `fullgraph`, they are built here.
        with timings.stage("triples"):
            pairs["disp3"] = TripleList.from_neighborlist(nbl_disp3)

    with timings.stage("load reference"):
        ref = _default_reference(structure.positions)
    stages = _stages(args, d3, param, damping, pairs, ref)
    three_body = damping.three is not None

    if args.compile:
        # A first run compiles every step (and its backward pass), so that
        # the timed run below shows the compiled steps only.
        with timings.stage("compile (warm-up)"):
            _evaluate(args, stages, structure, three_body, Timings(False))

    energy2, energy3, gradient = _evaluate(
        args, stages, structure, three_body, timings
    )

    print_system_info(args.structure, structure)
    if uses_native:
        print_native_build(_native.build_info())
    print_parameters(args.func, param, damping)
    if args.neighbor == "sparse":
        print_neighborlists(
            {
                term: nbl
                for term, nbl in lists.items()
                if isinstance(nbl, NeighborList)
            }
        )
    print_energy(
        energy2.detach(), None if energy3 is None else energy3.detach()
    )
    if gradient is not None:
        print_gradient(gradient, structure)
    timings.report()
