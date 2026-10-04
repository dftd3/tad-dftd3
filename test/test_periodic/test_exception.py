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
Invalid input for a periodic cell.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.typing import Tensor

from tad_dftd3 import damping, dftd3, disp
from tad_dftd3.cutoff import Cutoff

from ..cells import cells
from ..conftest import DEVICE

param = {
    "s6": torch.tensor(1.0000, dtype=torch.double),
    "s8": torch.tensor(1.2576, dtype=torch.double),
    "a1": torch.tensor(0.3768, dtype=torch.double),
    "a2": torch.tensor(4.5865, dtype=torch.double),
}

cutoff = Cutoff(cn=10.0, disp2=12.0)

numbers, positions, lattice = (t.to(DEVICE) for t in cells["diamond"])
periodic = torch.ones(3, dtype=torch.bool, device=DEVICE)


def test_shifts_without_lattice() -> None:
    shifts = build_periodic_shifts(lattice, periodic, cutoff.disp2)

    with pytest.raises(ValueError, match="no 'lattice'"):
        dftd3(
            Structure(numbers=numbers, positions=positions),
            param,
            shifts=shifts,
        )

    with pytest.raises(ValueError, match="no 'lattice'"):
        disp.dispersion2(
            Structure(numbers=numbers, positions=positions),
            param,
            torch.ones(8, 8),
            shifts=shifts,
        )


def test_vmap_shared_lattice() -> None:
    """
    `vmap` over a batched `Structure` splits every field, so one `(3, 3)`
    lattice shared by the batch would reach each system as one row of it.
    """
    batch = Structure(
        numbers=numbers.expand(3, -1),
        positions=positions.expand(3, -1, -1),
        lattice=lattice,
    )
    shifts = build_periodic_shifts(lattice, periodic, cutoff.disp2)

    def energy(structure: Structure) -> Tensor:
        return dftd3(structure, param, shifts=shifts, cutoff=cutoff)

    with pytest.raises(ValueError, match="own lattice and mask"):
        torch.func.vmap(energy)(batch)

    # also if only the lattice is stacked, but not the mask
    with pytest.raises(ValueError, match="own lattice and mask"):
        torch.func.vmap(energy)(batch.replace(lattice=lattice.expand(3, 3, 3)))

    # stacked per system, it works, and agrees with the batched call
    stacked = batch.replace(
        lattice=lattice.expand(3, 3, 3), periodic=periodic.expand(3, 3)
    )
    out = torch.func.vmap(energy)(stacked)
    assert torch.allclose(out, energy(batch), atol=1e-12, rtol=0)


def test_shifts_too_short() -> None:
    """A table built for a smaller cutoff would drop images."""
    shifts = build_periodic_shifts(lattice, periodic, cutoff.cn)

    with pytest.raises(ValueError, match="cutoff"):
        dftd3(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            param,
            shifts=shifts,
            cutoff=cutoff,
        )


def test_shifts_for_a_larger_cell() -> None:
    """A table built for a larger cell has too few image rings."""
    shifts = build_periodic_shifts(2.0 * lattice, periodic, cutoff.disp2)

    with pytest.raises(ValueError, match="image rings"):
        dftd3(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            param,
            shifts=shifts,
            cutoff=Cutoff(cn=cutoff.disp2, disp2=cutoff.disp2),
        )


def test_shifts_missing_an_axis() -> None:
    slab = torch.tensor([True, True, False], device=DEVICE)
    shifts = build_periodic_shifts(lattice, slab, cutoff.disp2)

    with pytest.raises(ValueError, match="periodic axis"):
        dftd3(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            param,
            shifts=shifts,
            cutoff=cutoff,
        )


@pytest.mark.parametrize("s9", [1.0, torch.tensor(1.0, dtype=torch.double)])
def test_three_body(s9: float | Tensor) -> None:
    """The three-body term has no periodic evaluation."""
    par = {**param, "s9": s9}

    with pytest.raises(ValueError, match="three-body"):
        dftd3(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            par,
        )

    with pytest.raises(ValueError, match="three-body"):
        disp.dispersion(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            par,
            torch.ones(8, 8),
        )

    with pytest.raises(ValueError, match="three-body"):
        disp.dispersion3(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            par,
            torch.ones(8, 8),
        )

    with pytest.raises(ValueError, match="three-body"):
        damping.dispersion_atm(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            torch.ones(8, 8),
        )


def test_three_body_zero_tensor_with_grad() -> None:
    """
    A zero tensor `s9` that is differentiated needs the three-body term
    (its derivative), so it is rejected as well.
    """
    s9 = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

    with pytest.raises(ValueError, match="three-body"):
        dftd3(
            Structure(numbers=numbers, positions=positions, lattice=lattice),
            {**param, "s9": s9},
        )
