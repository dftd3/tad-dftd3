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
The three-body term over triples enumerated beforehand
(:class:`tad_dftd3.sparse.TripleList`): the same energy as over the list it
was built from, and fixed-shape, so it survives
``torch.compile(fullgraph=True)``, ``vmap`` and ``jacrev``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import (
    NeighborList,
    build_neighborlist,
    build_neighborlists,
)
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import DampingParam, dftd3
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.sparse import TripleList, dispersion_atm_sparse

from ..cells import cells, param_on
from ..conftest import DEVICE, compile_test
from ..utils import load_structure

cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=12.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _param() -> dict[str, Tensor | float]:
    return {**param_on(DD64), "s9": torch.tensor(1.0, **DD64)}


def _structure(name: str) -> Structure:
    """A molecule, a batch of two or a triclinic cell."""
    if name == "molecule":
        return load_structure("other", "C6H5I-CH3SH", DD64)
    if name == "batch":
        return pack_structures(
            [
                load_structure("heavy28", "pbh4_bih3", DD64),
                load_structure("other", "C6H5I-CH3SH", DD64),
            ]
        )
    return cells["periodic_triclinic"].to(**DD64)


def _lists(structure: Structure) -> tuple[NeighborList, TripleList]:
    nbl = build_neighborlist(structure, cutoff.disp3)
    return nbl, TripleList.from_neighborlist(nbl)


@pytest.mark.parametrize("name", ["molecule", "batch", "cell"])
@pytest.mark.parametrize("max_triples", [50, 2_000_000])
def test_matches_the_list(name: str, max_triples: int) -> None:
    """In chunks or in one, the triples give the energy of their list."""
    structure = _structure(name)
    nbl, triples = _lists(structure)
    p = _param()

    want = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    got = dftd3(
        structure, p, cutoff=cutoff, nbl_disp3=triples, max_triples=max_triples
    )
    assert pytest.approx(want.cpu(), abs=tol) == got.cpu()


def test_wider_triples_match_the_list() -> None:
    """The triangles of a list with a skin, to be reused while the atoms
    move, reach beyond the cutoff and are masked at it."""
    structure = _structure("cell")
    nbl = build_neighborlist(structure, cutoff.disp3, skin=1.5)
    triples = TripleList.from_neighborlist(nbl)
    p = _param()

    exact = build_neighborlist(structure, cutoff.disp3)
    want = dftd3(structure, p, cutoff=cutoff, nbl_disp3=exact)
    got = dftd3(structure, p, cutoff=cutoff, nbl_disp3=triples)
    assert pytest.approx(want.cpu(), abs=tol) == got.cpu()


def _energy(
    structure: Structure, pairs: NeighborList | TripleList
) -> Callable[[Tensor], Tensor]:
    """The total energy as a function of the positions, with every term
    over a list built beforehand, as under a transform."""
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2)
    )
    p = _param()

    def energy(positions: Tensor) -> Tensor:
        return dftd3(
            structure.replace(positions=positions),
            p,
            cutoff=cutoff,
            nbl_cn=nbl_cn,
            nbl_disp2=nbl_disp2,
            nbl_disp3=pairs,
        ).sum()

    return energy


def _reference(
    structure: Structure, nbl: NeighborList
) -> tuple[Tensor, Tensor]:
    """Energy and gradient over the list, eagerly."""
    pos = structure.positions.detach().requires_grad_(True)
    energy = _energy(structure, nbl)(pos)
    (grad,) = torch.autograd.grad(energy, pos)
    return energy.detach(), grad


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.parametrize("name", ["molecule", "cell"])
def test_fullgraph_energy_and_gradient(name: str) -> None:
    """The three-body term over triples traces as one graph, and so does its
    gradient with respect to the positions."""
    structure = _structure(name)
    nbl, triples = _lists(structure)
    want, want_grad = _reference(structure, nbl)

    compiled = compile_test(_energy(structure, triples), fullgraph=True)
    pos = structure.positions.detach().requires_grad_(True)
    got = compiled(pos)
    (got_grad,) = torch.autograd.grad(got, pos)

    assert pytest.approx(want.item(), abs=tol) == got.item()
    assert pytest.approx(want_grad.cpu(), abs=tol) == got_grad.cpu()


@pytest.mark.parametrize("name", ["molecule", "cell"])
def test_jacrev_matches_the_list(name: str) -> None:
    """The gradient by `jacrev`, which the list cannot go through, is the
    one over the list by autograd."""
    structure = _structure(name)
    nbl, triples = _lists(structure)
    _, want = _reference(structure, nbl)

    got = torch.func.jacrev(_energy(structure, triples))(structure.positions)
    assert pytest.approx(want.cpu(), abs=tol) == got.cpu()


def test_vmap_over_s9() -> None:
    """With a tensor `s9` batched by `vmap`, only the three-body term is
    batched; each entry is the energy at that `s9`."""
    structure = _structure("molecule")
    _, triples = _lists(structure)
    s9 = torch.tensor([0.5, 1.0, 2.0], **DD64)

    def energy(s9: Tensor) -> Tensor:
        param = DampingParam(**{**param_on(DD64), "s9": s9})
        return dftd3(structure, param, cutoff=cutoff, nbl_disp3=triples)

    got = torch.func.vmap(energy)(s9)
    for value, row in zip(s9, got):
        assert pytest.approx(energy(value).cpu(), abs=tol) == row.cpu()


def test_no_triples_keeps_graph() -> None:
    """Without any triple, the energy is still a (zero) function of the
    positions, as over the list."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], **DD64)
    pos.requires_grad_(True)
    structure = Structure(
        numbers=torch.tensor([1, 1], device=DEVICE), positions=pos
    )
    _, triples = _lists(structure)
    atm = DampingParam(s9=1.0)

    energy = dispersion_atm_sparse(
        structure, torch.ones(2, 2, **DD64), atm, triples, cutoff=cutoff.disp3
    )
    (grad,) = torch.autograd.grad(energy.sum(), pos)

    assert triples.idx_i.numel() == 0
    assert (energy == 0).all() and (grad == 0).all()


def test_too_small_a_cutoff_is_an_error() -> None:
    structure = _structure("molecule")
    nbl = build_neighborlist(structure, cutoff.disp3 - 1.0)
    triples = TripleList.from_neighborlist(nbl)

    with pytest.raises(ValueError, match="smaller than the consumer"):
        dftd3(structure, _param(), cutoff=cutoff, nbl_disp3=triples)


def test_other_structure_is_an_error() -> None:
    _, triples = _lists(_structure("molecule"))

    with pytest.raises(ValueError, match="built for atoms of shape"):
        dftd3(_structure("batch"), _param(), cutoff=cutoff, nbl_disp3=triples)


@pytest.mark.parametrize("built_for", ["molecule", "cell"])
def test_periodicity_must_match(built_for: str) -> None:
    """Triples of a molecule do not hold the images of a cell, and the
    reverse."""
    cell = _structure("cell")
    molecule = Structure(numbers=cell.numbers, positions=cell.positions)
    source = cell if built_for == "cell" else molecule
    target = molecule if built_for == "cell" else cell
    _, triples = _lists(source)

    with pytest.raises(ValueError, match="lattice"):
        dftd3(target, _param(), cutoff=cutoff, nbl_disp3=triples)
