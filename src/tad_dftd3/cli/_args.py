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
Command line interface: arguments
=================================

The argument parser and the choices it offers.
"""

from __future__ import annotations

import argparse

from tad_mctc.cli._args import DTYPES, _positive_int

from .. import defaults

__all__ = ["DAMPING_VARIANTS", "DTYPES", "build_parser"]

DAMPING_VARIANTS: tuple[str, ...] = ("bj", "zero", "bjm", "zerom", "op", "cso")
"""Available ``--damping`` choices: the damping variants of the parameter
data base (see :func:`tad_dftd3.param.get_functional_params`)."""

_EPILOG = """\
The native CPU neighbour search of tad-mctc compiles with extra flags from
the TAD_MCTC_NATIVE_CFLAGS environment variable, e.g.
TAD_MCTC_NATIVE_CFLAGS=-mavx2 tad_dftd3 --func pbe0 structure.xyz. A build
with them runs only on CPUs that support them. The "Native extension"
section of the output lists the flags of the build in use.
"""


def _non_negative_float(value: str) -> float:
    """An ``argparse`` type for a distance in Bohr, which must not be
    negative."""
    try:
        number = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid float value: '{value}'"
        ) from e
    if number < 0.0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """The parser of the ``tad_dftd3`` command line."""
    parser = argparse.ArgumentParser(
        prog="tad_dftd3",
        description="Calculate the DFT-D3 dispersion energy of a structure.",
        epilog=_EPILOG,
    )
    parser.add_argument(
        "structure",
        type=str,
        help="Path to the structure file (xyz, Turbomole coord, ...).",
    )
    parser.add_argument(
        "--func",
        dest="func",
        required=True,
        help=(
            "Functional whose damping parameters are used, e.g. 'pbe0' "
            "(case- and hyphen-insensitive, with the aliases of s-dftd3)."
        ),
    )
    parser.add_argument(
        "--damping",
        dest="damping",
        choices=DAMPING_VARIANTS,
        default=None,
        help=(
            "Damping variant of the parameters. Defaults to the first of "
            "'bj' and 'zero' the functional has."
        ),
    )
    parser.add_argument(
        "--no-atm",
        dest="atm",
        action="store_false",
        help="Leave out the three-body (Axilrod-Teller-Muto) term.",
    )
    parser.add_argument(
        "--grad",
        action="store_true",
        help=(
            "Also compute the gradient of the energy with respect to the "
            "positions, in one backward pass."
        ),
    )
    parser.add_argument(
        "--neighbor",
        dest="neighbor",
        choices=("dense", "sparse"),
        default="sparse",
        help=(
            "How the pairs of every term are enumerated: 'dense' evaluates "
            "all atom pairs (and triples), masked at each term's cutoff, "
            "for a cell with every periodic image; 'sparse' builds a padded "
            "neighbour list per term first, in one shared search. Defaults "
            "to 'sparse'."
        ),
    )
    parser.add_argument(
        "--nlist-only",
        dest="nlist_only",
        action="store_true",
        help=(
            "Only build the neighbour lists and print a summary of them, "
            "skipping the energy. Incompatible with '--neighbor dense', "
            "which builds no list."
        ),
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=("graph", "recompute"),
        default="graph",
        help=(
            "How the backward pass of '--grad' gets the intermediates: "
            "'graph' keeps those of every pair and triple (memory "
            "O(n_pairs)); 'recompute' checkpoints them in chunks, over a "
            "neighbour list and the triples of a cell (memory O(chunk)). "
            "Defaults to 'graph'."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile every step with torch.compile(fullgraph=True), in a "
            "first, timed warm-up run (seconds, once per structure size), "
            "and report the compiled run. Pays off for large systems. The "
            "sparse three-body term is then evaluated over its triples, "
            "enumerated beforehand. Incompatible with '--mode recompute'."
        ),
    )
    parser.add_argument(
        "--max-triples",
        dest="max_triples",
        type=_positive_int,
        default=2_000_000,
        metavar="N",
        help=(
            "Upper bound on the triples of the three-body term held in "
            "memory at once, for a cell or over a neighbour list. Defaults "
            "to 2000000."
        ),
    )
    for term, name, default in (
        ("cn", "coordination number", defaults.D3_CN_CUTOFF),
        ("disp2", "two-body term", defaults.D3_DISP2_CUTOFF),
        ("disp3", "three-body term", defaults.D3_DISP3_CUTOFF),
    ):
        parser.add_argument(
            f"--cutoff-{term}",
            dest=f"cutoff_{term}",
            type=_non_negative_float,
            default=default,
            metavar="BOHR",
            help=(
                f"Real-space cutoff of the {name}, in Bohr. Defaults to "
                f"{default}."
            ),
        )
    for term, name, default in (
        ("2", "two-body term", defaults.D3_DISP2_WIDTH),
        ("3", "three-body term", defaults.D3_DISP3_WIDTH),
    ):
        parser.add_argument(
            f"--width{term}",
            dest=f"width{term}",
            type=_non_negative_float,
            default=default,
            metavar="BOHR",
            help=(
                f"Width of the smooth cutoff of the {name}, in Bohr: the "
                "term is switched off over this distance below its cutoff. "
                f"Defaults to {default}, the hard cutoff."
            ),
        )
    parser.add_argument(
        "--coldfusion-check",
        dest="coldfusion_check",
        action="store_true",
        help=(
            "Run the interatomic-distance sanity check while reading. "
            "That check builds its own neighbour list on the CPU, before "
            "'--cuda' takes effect, and can dominate read time for a "
            "large structure -- opt in for an untrusted geometry. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Print the wall time of each computation step, including "
            "each step of the neighbour-list build."
        ),
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        choices=sorted(DTYPES),
        default="float64",
        help=(
            "Floating point precision of the positions, lattice and "
            "parameters. Defaults to 'float64'."
        ),
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run on the first CUDA device instead of the CPU.",
    )
    parser.add_argument(
        "--omp",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Number of threads torch uses for intra-op CPU parallelism "
            "during this run."
        ),
    )
    return parser
