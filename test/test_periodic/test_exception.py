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

from ..cells import cells, param
from ..conftest import DEVICE

cutoff = Cutoff(cn=10.0, disp2=12.0)

cell = cells["diamond"].to(DEVICE)
molecule = Structure(numbers=cell.numbers, positions=cell.positions)
assert cell.lattice is not None and cell.periodic is not None
lattice, periodic = cell.lattice, cell.periodic

c6 = torch.ones(8, 8, dtype=torch.double, device=DEVICE)


def test_shifts_without_lattice() -> None:
    shifts = build_periodic_shifts(lattice, periodic, cutoff.disp2)

    with pytest.raises(ValueError, match="lattice"):
        dftd3(molecule, param, shifts=shifts)

    with pytest.raises(ValueError, match="lattice"):
        disp.dispersion2(molecule, param, c6, pairs=shifts)


def test_vmap_shared_lattice() -> None:
    """
    `vmap` over a batched `Structure` splits every field, so one `(3, 3)`
    lattice shared by the batch would reach each system as one row of it,
    which fails rather than giving numbers.
    """
    batch = Structure(
        numbers=cell.numbers.expand(3, -1),
        positions=cell.positions.expand(3, -1, -1),
        lattice=lattice,
    )
    shifts = build_periodic_shifts(lattice, periodic, cutoff.disp2)

    def energy(structure: Structure) -> Tensor:
        return dftd3(structure, param, shifts=shifts, cutoff=cutoff)

    with pytest.raises(IndexError):
        torch.func.vmap(energy)(batch)

    # also if only the lattice is stacked, but not the mask
    with pytest.raises(IndexError):
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
        dftd3(cell, param, shifts=shifts, cutoff=cutoff)


def test_shifts_for_a_larger_cell() -> None:
    """A table built for a larger cell has too few image rings."""
    shifts = build_periodic_shifts(2.0 * lattice, periodic, cutoff.disp2)
    same = Cutoff(cn=cutoff.disp2, disp2=cutoff.disp2)

    with pytest.raises(ValueError, match="image rings"):
        dftd3(cell, param, shifts=shifts, cutoff=same)


def test_shifts_missing_an_axis() -> None:
    slab = torch.tensor([True, True, False], device=DEVICE)
    shifts = build_periodic_shifts(lattice, slab, cutoff.disp2)

    with pytest.raises(ValueError, match="periodic axis"):
        dftd3(cell, param, shifts=shifts, cutoff=cutoff)


@pytest.mark.parametrize("s9", [1.0, torch.tensor(1.0, dtype=torch.double)])
def test_three_body_is_evaluated(s9: float | Tensor) -> None:
    """The three-body term of a cell is part of every entry point."""
    par = {**param, "s9": s9}
    cut = Cutoff(cn=10.0, disp2=12.0, disp3=8.0)

    energy = dftd3(cell, par, cutoff=cut)
    off = dftd3(cell, {**param, "s9": 0.0}, cutoff=cut)
    assert not torch.allclose(energy, off, atol=1e-10, rtol=0)

    c6 = disp.D3Model(cutoff=cut).c6(cell)
    e3 = disp.dispersion3(cell, par, c6, cutoff=cut)
    assert pytest.approx(e3.cpu(), abs=1e-12) == (energy - off).cpu()
    assert (
        pytest.approx(e3.cpu(), abs=1e-12)
        == damping.dispersion_atm_periodic(
            cell, c6, damping.DampingParam(s9=s9), cutoff=cut.disp3
        ).cpu()
    )
    assert (
        pytest.approx(energy.cpu(), abs=1e-12)
        == disp.dispersion(cell, par, c6, cutoff=cut).cpu()
    )


def test_three_body_zero_tensor_with_grad() -> None:
    """
    A zero tensor `s9` that is differentiated needs the three-body term (its
    derivative), which for a cell is its energy.
    """
    cut = Cutoff(cn=10.0, disp2=12.0, disp3=8.0)
    s9 = torch.tensor(0.0, dtype=torch.double, requires_grad=True)

    energy = dftd3(cell, {**param, "s9": s9}, cutoff=cut)
    (grad,) = torch.autograd.grad(energy.sum(), s9)

    off = dftd3(cell, {**param, "s9": 0.0}, cutoff=cut)
    full = dftd3(cell, {**param, "s9": 1.0}, cutoff=cut)
    assert pytest.approx(grad.item(), abs=1e-12) == (full - off).sum().item()
