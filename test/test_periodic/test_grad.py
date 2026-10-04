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
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import cells, random_cell
from ..conftest import DEVICE, FAST_MODE
from ..reference import reference_gradient, reference_hessian

param = {
    "s6": torch.tensor(1.0000, dtype=torch.double),
    "s8": torch.tensor(1.2576, dtype=torch.double),
    "a1": torch.tensor(0.3768, dtype=torch.double),
    "a2": torch.tensor(4.5865, dtype=torch.double),
}

cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-11

TRICLINIC = torch.tensor(
    [[7.0, 0.0, 0.0], [1.2, 6.5, 0.0], [0.6, 0.9, 6.0]], dtype=torch.double
)


def _param(dd: DD) -> dict[str, Tensor | float]:
    return {k: v.to(**dd) for k, v in param.items()}


def _gradients(
    numbers: Tensor,
    positions: Tensor,
    lattice: Tensor,
    periodic: Tensor | None = None,
    cut: Cutoff | None = cutoff,
) -> tuple[Tensor, Tensor]:
    """Derivatives of the energy with respect to positions and lattice."""
    positions = positions.clone().requires_grad_(True)
    lattice = lattice.clone().requires_grad_(True)

    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    energy = dftd3(
        Structure(
            numbers=numbers,
            positions=positions,
            lattice=lattice,
            periodic=periodic,
        ),
        _param(dd),
        cutoff=cut,
    )
    grad_pos, grad_lat = torch.autograd.grad(energy.sum(), (positions, lattice))
    return grad_pos, grad_lat


def _virial(
    positions: Tensor, lattice: Tensor, grad_pos: Tensor, grad_lat: Tensor
) -> Tensor:
    """Derivative of the energy with respect to a homogeneous strain."""
    return positions.mT @ grad_pos + lattice.mT @ grad_lat


@pytest.mark.parametrize("name", list(cells))
def test_gradient_and_virial(name: str) -> None:
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    ref_grad, ref_virial = reference_gradient(
        numbers, positions, param, cutoff=cutoff, lattice=lattice
    )
    grad_pos, grad_lat = _gradients(numbers, positions, lattice)

    assert pytest.approx(ref_grad.cpu(), abs=tol, rel=0) == grad_pos.cpu()

    virial = _virial(positions, lattice, grad_pos, grad_lat)
    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()


@pytest.mark.parametrize("name", ["urea", "diamond"])
def test_gradient_and_virial_default_cutoffs(name: str) -> None:
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    ref_grad, ref_virial = reference_gradient(
        numbers, positions, param, lattice=lattice
    )
    grad_pos, grad_lat = _gradients(numbers, positions, lattice, cut=None)

    assert pytest.approx(ref_grad.cpu(), abs=tol, rel=0) == grad_pos.cpu()

    virial = _virial(positions, lattice, grad_pos, grad_lat)
    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()


@pytest.mark.parametrize(
    "periodic", [[True, True, False], [True, False, False]]
)
def test_gradient_and_virial_low_dimensional(periodic: list[bool]) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = random_cell(TRICLINIC, 5, dd, seed=11)
    mask = torch.tensor(periodic, device=DEVICE)

    ref_grad, ref_virial = reference_gradient(
        numbers,
        positions,
        param,
        cutoff=cutoff,
        lattice=lattice,
        periodic=mask,
    )
    grad_pos, grad_lat = _gradients(numbers, positions, lattice, mask)

    assert pytest.approx(ref_grad.cpu(), abs=tol, rel=0) == grad_pos.cpu()

    virial = _virial(positions, lattice, grad_pos, grad_lat)
    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()

    # no image along an open axis, so no derivative by its lattice vector
    open_axes = ~mask.cpu()
    assert (grad_lat.cpu()[open_axes] == 0).all()


def test_gradient_unwrapped_positions() -> None:
    """Folding into the cell adds whole lattice vectors, so moving an atom
    by one changes neither the gradient nor the lattice derivative."""
    numbers, positions, lattice = (t.to(DEVICE) for t in cells["urea"])

    generator = torch.Generator().manual_seed(1)
    offsets = torch.randint(-4, 5, positions.shape, generator=generator)
    unwrapped = positions + offsets.to(positions) @ lattice

    grad_pos, grad_lat = _gradients(numbers, positions, lattice)
    grad_pos_unwrapped, grad_lat_unwrapped = _gradients(
        numbers, unwrapped, lattice
    )
    assert pytest.approx(grad_pos.cpu(), abs=tol) == grad_pos_unwrapped.cpu()

    # The energy of the unwrapped geometry depends on the lattice also
    # through the offsets, which the strain derivative accounts for.
    ref_grad, ref_virial = reference_gradient(
        numbers, unwrapped, param, cutoff=cutoff, lattice=lattice
    )
    assert pytest.approx(ref_grad.cpu(), abs=tol) == grad_pos_unwrapped.cpu()

    virial = _virial(unwrapped, lattice, grad_pos_unwrapped, grad_lat_unwrapped)
    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()

    virial_wrapped = _virial(positions, lattice, grad_pos, grad_lat)
    assert pytest.approx(virial_wrapped.cpu(), abs=tol) == virial.cpu()


@pytest.mark.parametrize("name", ["periodic_triclinic", "diamond"])
@pytest.mark.parametrize("mode", ["rev-rev", "fwd-rev"])
def test_hessian(name: str, mode: str) -> None:
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    def energy(pos: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=numbers, positions=pos, lattice=lattice),
            _param(dd),
            cutoff=cutoff,
        ).sum()

    if mode == "rev-rev":
        hessian = torch.func.jacrev(torch.func.jacrev(energy))(positions)
    else:
        hessian = torch.func.hessian(energy)(positions)

    ref = reference_hessian(
        numbers, positions, param, cutoff=cutoff, lattice=lattice
    )
    assert pytest.approx(ref.cpu(), abs=1e-10, rel=0) == hessian.cpu()


def test_batch_padding() -> None:
    """
    In a padded batch of cells, the derivatives are those of each cell
    alone, finite, and exactly zero for the padding atoms.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    systems = [
        tuple(t.to(DEVICE) for t in cells["periodic_triclinic"]),
        random_cell(TRICLINIC, 3, dd, seed=12),
    ]
    numbers = pack([s[0] for s in systems])
    positions = pack([s[1] for s in systems])
    lattice = torch.stack([s[2] for s in systems])

    grad_pos, grad_lat = _gradients(numbers, positions, lattice)

    assert torch.isfinite(grad_pos).all()
    assert torch.isfinite(grad_lat).all()
    for i, (n, p, lat) in enumerate(systems):
        nat = n.shape[-1]
        ref_pos, ref_lat = _gradients(n, p, lat)
        assert pytest.approx(ref_pos.cpu(), abs=tol) == grad_pos[i, :nat].cpu()
        assert pytest.approx(ref_lat.cpu(), abs=tol) == grad_lat[i].cpu()
        assert (grad_pos[i, nat:] == 0).all()


def _gradcheck_setup() -> tuple[Tensor, Tensor, Tensor]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    lattice = 5.0 * torch.eye(3, dtype=torch.double)
    lattice = lattice + 0.3 * torch.rand(
        3, 3, generator=torch.Generator().manual_seed(0), dtype=torch.double
    )
    return random_cell(lattice, 3, dd, seed=13)


@pytest.mark.grad
def test_gradcheck() -> None:
    """Positions and lattice together, against finite differences."""
    numbers, positions, lattice = _gradcheck_setup()
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    small = Cutoff(cn=8.0, disp2=10.0)

    def func(pos: Tensor, lat: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=numbers, positions=pos, lattice=lat),
            _param(dd),
            cutoff=small,
        )

    inputs = (
        positions.clone().requires_grad_(True),
        lattice.clone().requires_grad_(True),
    )
    assert dgradcheck(func, inputs, fast_mode=FAST_MODE)


@pytest.mark.grad
def test_gradgradcheck() -> None:
    numbers, positions, lattice = _gradcheck_setup()
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    small = Cutoff(cn=8.0, disp2=10.0)

    def func(pos: Tensor, lat: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=numbers, positions=pos, lattice=lat),
            _param(dd),
            cutoff=small,
        )

    inputs = (
        positions.clone().requires_grad_(True),
        lattice.clone().requires_grad_(True),
    )
    assert dgradgradcheck(func, inputs, fast_mode=FAST_MODE)
