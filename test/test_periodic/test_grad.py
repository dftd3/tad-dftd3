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
Derivatives of the dense periodic dispersion energy, against s-dftd3.

s-dftd3 gives the nuclear gradient, the virial and the Hessian with respect
to the positions. The derivative with respect to the lattice enters the
virial, ``positions.T @ dE/dpositions + lattice.T @ dE/dlattice`` (see
:func:`test.reference.reference_gradient`).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.io.structure import pack_structures
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import TRICLINIC, cells, param, param_on, random_cell
from ..conftest import DEVICE, FAST_MODE
from ..reference import reference_gradient, reference_hessian

cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-11

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _gradients(
    structure: Structure, cut: Cutoff | None = cutoff
) -> tuple[Tensor, Tensor]:
    """Derivatives of the energy with respect to positions and lattice."""
    assert structure.lattice is not None
    positions = structure.positions.clone().requires_grad_(True)
    lattice = structure.lattice.clone().requires_grad_(True)

    cell = structure.replace(positions=positions, lattice=lattice)
    energy = dftd3(cell, param_on(DD64), cutoff=cut)
    grad_pos, grad_lat = torch.autograd.grad(energy.sum(), (positions, lattice))
    return grad_pos, grad_lat


def _check(
    structure: Structure, cut: Cutoff | None = cutoff
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Check the gradient and the virial of `structure` against s-dftd3 at the
    same cutoffs. Returns the derivatives with respect to the positions and
    the lattice, and the virial.

    The virial, the derivative with respect to a homogeneous strain, is
    ``positions.T @ dE/dpositions + lattice.T @ dE/dlattice``.
    """
    assert structure.lattice is not None
    ref_grad, ref_virial = reference_gradient(structure, param, cutoff=cut)
    grad_pos, grad_lat = _gradients(structure, cut)

    assert pytest.approx(ref_grad.cpu(), abs=tol, rel=0) == grad_pos.cpu()

    virial = structure.positions.mT @ grad_pos + structure.lattice.mT @ grad_lat
    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()
    return grad_pos, grad_lat, virial


@pytest.mark.parametrize("name", list(cells))
def test_gradient_and_virial(name: str) -> None:
    _check(cells[name].to(DEVICE))


@pytest.mark.parametrize("name", ["urea", "diamond"])
def test_gradient_and_virial_default_cutoffs(name: str) -> None:
    _check(cells[name].to(DEVICE), cut=None)


@pytest.mark.parametrize(
    "periodic", [[True, True, False], [True, False, False]]
)
def test_gradient_and_virial_low_dimensional(periodic: list[bool]) -> None:
    mask = torch.tensor(periodic, device=DEVICE)
    _, grad_lat, _ = _check(
        random_cell(TRICLINIC, 5, DD64, seed=11, periodic=mask)
    )

    # no image along an open axis, so no derivative by its lattice vector
    assert (grad_lat.cpu()[~mask.cpu()] == 0).all()


def test_gradient_unwrapped_positions() -> None:
    """
    Folding into the cell adds whole lattice vectors, so moving an atom by
    one changes neither the gradient nor the virial. The energy of the
    unwrapped geometry depends on the lattice also through the offsets,
    which the strain derivative accounts for.
    """
    structure = cells["urea"].to(DEVICE)
    assert structure.lattice is not None

    generator = torch.Generator().manual_seed(1)
    offsets = torch.randint(
        -4, 5, structure.positions.shape, generator=generator
    )
    unwrapped = structure.positions + offsets.to(**DD64) @ structure.lattice
    moved = structure.replace(positions=unwrapped)

    grad_pos, _, virial = _check(structure)
    grad_pos_unwrapped, _, virial_unwrapped = _check(moved)
    assert pytest.approx(grad_pos.cpu(), abs=tol) == grad_pos_unwrapped.cpu()
    assert pytest.approx(virial.cpu(), abs=tol) == virial_unwrapped.cpu()


@pytest.mark.parametrize("name", ["periodic_triclinic", "diamond"])
@pytest.mark.parametrize("mode", ["rev-rev", "fwd-rev"])
def test_hessian(name: str, mode: str) -> None:
    structure = cells[name].to(DEVICE)

    def energy(pos: Tensor) -> Tensor:
        cell = structure.replace(positions=pos)
        return dftd3(cell, param_on(DD64), cutoff=cutoff).sum()

    if mode == "rev-rev":
        hessian = torch.func.jacrev(torch.func.jacrev(energy))
    else:
        hessian = torch.func.hessian(energy)

    ref = reference_hessian(structure, param, cutoff=cutoff)
    out = hessian(structure.positions)
    assert pytest.approx(ref.cpu(), abs=1e-10, rel=0) == out.cpu()


def test_batch_padding() -> None:
    """
    In a padded batch of cells, the derivatives are those of each cell
    alone, finite, and exactly zero for the padding atoms.
    """
    systems = [
        cells["periodic_triclinic"].to(DEVICE),
        random_cell(TRICLINIC, 3, DD64, seed=12),
    ]
    grad_pos, grad_lat = _gradients(pack_structures(systems))

    assert torch.isfinite(grad_pos).all()
    assert torch.isfinite(grad_lat).all()
    for i, system in enumerate(systems):
        nat = system.numbers.shape[-1]
        ref_pos, ref_lat = _gradients(system)
        assert pytest.approx(ref_pos.cpu(), abs=tol) == grad_pos[i, :nat].cpu()
        assert pytest.approx(ref_lat.cpu(), abs=tol) == grad_lat[i].cpu()
        assert (grad_pos[i, nat:] == 0).all()


def _gradcheck_inputs() -> tuple[Structure, tuple[Tensor, Tensor]]:
    """A small distorted cell, and its positions and lattice as leaves."""
    lattice = 5.0 * torch.eye(3, dtype=torch.double)
    lattice = lattice + 0.3 * torch.rand(
        3, 3, generator=torch.Generator().manual_seed(0), dtype=torch.double
    )
    structure = random_cell(lattice, 3, DD64, seed=13)
    assert structure.lattice is not None

    inputs = (
        structure.positions.clone().requires_grad_(True),
        structure.lattice.clone().requires_grad_(True),
    )
    return structure, inputs


def _energy_fn(structure: Structure) -> Callable[[Tensor, Tensor], Tensor]:
    small = Cutoff(cn=8.0, disp2=10.0)

    def func(pos: Tensor, lat: Tensor) -> Tensor:
        cell = structure.replace(positions=pos, lattice=lat)
        return dftd3(cell, param_on(DD64), cutoff=small)

    return func


@pytest.mark.grad
def test_gradcheck() -> None:
    """Positions and lattice together, against finite differences."""
    structure, inputs = _gradcheck_inputs()
    assert dgradcheck(_energy_fn(structure), inputs, fast_mode=FAST_MODE)


@pytest.mark.grad
def test_gradgradcheck() -> None:
    structure, inputs = _gradcheck_inputs()
    assert dgradgradcheck(_energy_fn(structure), inputs, fast_mode=FAST_MODE)
