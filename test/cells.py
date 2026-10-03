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
Periodic cells for the tests of the periodic dispersion energy.

Real crystals from :func:`tad_mctc.data.structures.get_structure` (molecular
crystals of the X23 set, and a few small inorganic and synthetic cells),
plus :func:`random_cell` for shapes no real crystal here has: cells small
enough that an atom interacts with its own images, and slabs and chains.
"""

from __future__ import annotations

import torch
from tad_mctc.data.structures import get_structure
from tad_mctc.typing import DD, Tensor

__all__ = ["Cell", "cells", "random_cell"]


Cell = tuple[Tensor, Tensor, Tensor]
"""``(numbers, positions, lattice)``, all periodic along every axis."""

_SOURCES: dict[str, tuple[str, str]] = {
    "ammonia": ("x23", "ammonia"),
    "urea": ("x23", "urea"),
    "formamide": ("x23", "formamide"),
    "diamond": ("other", "diamond"),
    "nacl": ("other", "nacl"),
    "periodic_cubic": ("other", "periodic_cubic"),
    "periodic_triclinic": ("other", "periodic_triclinic"),
    "periodic_one_atom": ("other", "periodic_one_atom"),
}
"""Name used in the tests -> ``(collection, record)``."""


def _cell(collection: str, record: str) -> Cell:
    structure = get_structure(collection, record, dtype=torch.double)
    assert structure.lattice is not None
    return structure.numbers, structure.positions, structure.lattice


cells: dict[str, Cell] = {
    name: _cell(*source) for name, source in _SOURCES.items()
}
"""Crystals, in double precision on the CPU."""


def random_cell(lattice: Tensor, nat: int, dd: DD, seed: int = 0) -> Cell:
    """
    ``nat`` atoms (Li to Zn) at random fractional coordinates of
    `lattice`, so every atom lies inside the cell whatever its shape.

    Returns ``(numbers, positions, lattice)``, with `lattice` cast to `dd`.
    """
    lattice = lattice.to(**dd)

    generator = torch.Generator().manual_seed(seed)
    fractional = torch.rand(nat, 3, generator=generator, dtype=torch.double)
    positions = fractional.to(**dd) @ lattice

    numbers = torch.randint(3, 31, (nat,), generator=generator)
    return numbers.to(dd["device"]), positions, lattice
