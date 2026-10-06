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
Command line interface: output
==============================

The sections the tool prints after a run, besides the summary of the
structure and of the native extension, which are tad-mctc's.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import real_atoms
from tad_mctc.cli._output import _describe_atom
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.typing import Tensor

from ..damping import Damping, DampingParam

__all__ = [
    "print_energy",
    "print_gradient",
    "print_neighborlists",
    "print_parameters",
]

_PARAM_NAMES = (
    "s6",
    "s8",
    "s9",
    "a1",
    "a2",
    "a3",
    "a4",
    "rs6",
    "rs8",
    "rs9",
    "alp",
    "bet",
    "sxc",
    "rsxc",
)
"""The numeric fields of :class:`~tad_dftd3.damping.DampingParam`."""


def _format_values(values: Tensor, fmt: str) -> str:
    """A quantity of a structure, one value per frame of a batch."""
    return " ".join(f"{v:{fmt}}" for v in values.flatten().tolist())


def print_parameters(func: str, param: DampingParam, damping: Damping) -> None:
    """Print the functional, the damping and the parameters that are set."""
    three = "none" if damping.three is None else type(damping.three).__name__

    print()
    print("Damping")
    print("-------")
    print(f"  functional  {func}")
    print(f"  damping     {param.damping}")
    print(f"  two-body    {type(damping.two).__name__}")
    print(f"  three-body  {three}")
    for name in _PARAM_NAMES:
        value = getattr(param, name)
        if value is not None:
            print(f"  {name:<10}  {float(value):.6f}")


def print_neighborlists(lists: dict[str, NeighborList]) -> None:
    """Print the size of the neighbour list of every term, one row each."""
    print()
    print("Neighbour lists")
    print("---------------")
    print(
        f"  {'term':<6}  {'cutoff':>8}  {'pairs':>12}  {'capacity':>12}  "
        "overflow"
    )
    for term, nbl in lists.items():
        # `sum` would first copy the boolean mask to int64, 8 bytes per slot.
        n_pairs = int(nbl.mask.count_nonzero())
        print(
            f"  {term:<6}  {nbl.cutoff:>8.4f}  {n_pairs:>12}  "
            f"{nbl.mask.numel():>12}  {'yes' if nbl.overflow else 'no'}"
        )


def print_energy(energy2: Tensor, energy3: Tensor | None) -> None:
    """Print the dispersion energy of the structure, per frame of a batch,
    split into the two- and three-body term. The arguments are the
    atom-resolved energies, zero on padding atoms."""
    e2 = energy2.sum(-1)
    total = e2 if energy3 is None else e2 + energy3.sum(-1)

    print()
    print("Results")
    print("-------")
    print(f"  two-body    {_format_values(e2, ' .12f')} Eh")
    if energy3 is not None:
        print(f"  three-body  {_format_values(energy3.sum(-1), ' .12f')} Eh")
    print(f"  total       {_format_values(total, ' .12f')} Eh")


def print_gradient(gradient: Tensor, structure: Structure) -> None:
    """Print summary statistics of the gradient with respect to the
    positions instead of the raw tensor: its norm per frame and its largest
    atomic force over all real atoms."""
    is_real = real_atoms(structure.numbers)
    # The norm of the force on each atom, (..., nat).
    per_atom = torch.linalg.vector_norm(gradient, dim=-1)
    real_index = is_real.nonzero()
    imax = int(torch.argmax(per_atom[is_real]))

    print()
    print("Gradient")
    print("--------")
    norm = torch.linalg.vector_norm(gradient, dim=(-2, -1))
    print(f"  norm        {_format_values(norm, ' .12f')} Eh/a0")
    print(
        f"  max atom    {per_atom[is_real][imax].item():.12f} Eh/a0  "
        f"({_describe_atom(structure.numbers, real_index[imax])})"
    )
