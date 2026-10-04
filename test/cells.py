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
Periodic cells and parameters for the tests of the periodic dispersion
energy.

Real crystals from :func:`tad_mctc.data.structures.get_structure` (molecular
crystals of the X23 set, and a few small inorganic and synthetic cells),
plus :func:`random_cell` for shapes no real crystal here has: cells small
enough that an atom interacts with its own images, and slabs and chains.
"""

from __future__ import annotations

import torch
from tad_mctc import Structure
from tad_mctc.data.structures import get_structure
from tad_mctc.typing import DD, Tensor

__all__ = ["TRICLINIC", "cells", "param", "param_on", "random_cell"]


param: dict[str, Tensor | float] = {
    "s6": torch.tensor(1.0000, dtype=torch.double),
    "s8": torch.tensor(1.2576, dtype=torch.double),
    "a1": torch.tensor(0.3768, dtype=torch.double),
    "a2": torch.tensor(4.5865, dtype=torch.double),
}
"""TPSS0-D3(BJ), without the three-body term, which is not periodic."""


def param_on(dd: DD) -> dict[str, Tensor | float]:
    """:data:`param` on the device and dtype of `dd`."""
    return {
        k: v.to(**dd) if isinstance(v, Tensor) else v for k, v in param.items()
    }


TRICLINIC = torch.tensor(
    [[7.0, 0.0, 0.0], [1.2, 6.5, 0.0], [0.6, 0.9, 6.0]], dtype=torch.double
)
"""Lattice of a triclinic cell, in Bohr."""


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


cells: dict[str, Structure] = {
    name: get_structure(collection, record, dtype=torch.double)
    for name, (collection, record) in _SOURCES.items()
}
"""Crystals, periodic along every axis, in double precision on the CPU."""


def random_cell(
    lattice: Tensor,
    nat: int,
    dd: DD,
    seed: int = 0,
    periodic: Tensor | None = None,
) -> Structure:
    """
    ``nat`` atoms (Li to Zn) at random fractional coordinates of
    `lattice`, so every atom lies inside the cell whatever its shape.
    `periodic` defaults to all three axes.
    """
    lattice = lattice.to(**dd)

    generator = torch.Generator().manual_seed(seed)
    fractional = torch.rand(nat, 3, generator=generator, dtype=torch.double)
    positions = fractional.to(**dd) @ lattice

    numbers = torch.randint(3, 31, (nat,), generator=generator)
    return Structure(
        numbers=numbers.to(dd["device"]),
        positions=positions,
        lattice=lattice,
        periodic=periodic,
    )
