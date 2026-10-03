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
Composition of `torch.func` transforms (`vmap`, `jacrev`, `jacfwd`) with the
dense periodic energy, with respect to the positions and the lattice.

The periodic image shifts are built once for the whole batch beforehand,
since building them reads the lattice. Batched results are compared to the
same quantity from plain autograd, cell by cell.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import data, dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import cells, random_cell
from ..conftest import DEVICE

tol = 1e-10

param = {
    "s6": torch.tensor(1.0000, dtype=torch.double),
    "s8": torch.tensor(1.2576, dtype=torch.double),
    "a1": torch.tensor(0.3768, dtype=torch.double),
    "a2": torch.tensor(4.5865, dtype=torch.double),
}

cutoff = Cutoff(cn=10.0, disp2=12.0)

TRICLINIC = torch.tensor(
    [[7.0, 0.0, 0.0], [1.2, 6.5, 0.0], [0.6, 0.9, 6.0]], dtype=torch.double
)

Batch = tuple[Tensor, Tensor, Tensor, Tensor, PeriodicShifts]


def setup(mixed_periodicity: bool = False) -> Batch:
    """
    Two cells of different shape and size, the smaller one padded:
    ``(numbers, positions, lattice, periodic, shifts)``.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    systems = [
        tuple(t.to(DEVICE) for t in cells["periodic_triclinic"]),
        random_cell(TRICLINIC, 3, dd, seed=21),
    ]

    numbers = pack([s[0] for s in systems])
    positions = pack([s[1] for s in systems])
    lattice = torch.stack([s[2] for s in systems])

    periodic = torch.ones(2, 3, dtype=torch.bool, device=DEVICE)
    if mixed_periodicity:
        periodic[1, 2] = False

    shifts = build_periodic_shifts(
        lattice, periodic, max(cutoff.cn, cutoff.disp2)
    )
    return numbers, positions, lattice, periodic, shifts


def energy_fn(shifts: PeriodicShifts) -> Callable[..., Tensor]:
    """Total energy of one cell, ``(numbers, positions, lattice, periodic)``."""

    def energy(n: Tensor, p: Tensor, lat: Tensor, per: Tensor) -> Tensor:
        return dftd3(
            n,
            p,
            param,
            lattice=lat,
            periodic=per,
            shifts=shifts,
            cutoff=cutoff,
        ).sum()

    return energy


def per_cell(fn: Callable[..., Tensor], batch: Batch) -> list[Tensor]:
    """
    `fn` of each cell alone, evaluated in plain eager mode with the padding
    atoms (zero) still in place, so the shapes match the batched result.
    """
    numbers, positions, lattice, periodic, _ = batch
    return [
        fn(numbers[i], positions[i], lattice[i], periodic[i])
        for i in range(numbers.shape[0])
    ]


def autograd_gradients(batch: Batch) -> list[tuple[Tensor, Tensor]]:
    """
    Reference derivatives with respect to positions and lattice, by plain
    autograd without any transform, cell by cell.
    """
    numbers, positions, lattice, periodic, shifts = batch
    energy = energy_fn(shifts)

    gradients = []
    for i in range(numbers.shape[0]):
        pos = positions[i].clone().requires_grad_(True)
        lat = lattice[i].clone().requires_grad_(True)
        e = energy(numbers[i], pos, lat, periodic[i])

        grad_pos, grad_lat = torch.autograd.grad(e, (pos, lat))
        gradients.append((grad_pos, grad_lat))

    return gradients


@pytest.mark.parametrize("mixed_periodicity", [False, True])
def test_vmap_energy(mixed_periodicity: bool) -> None:
    batch = setup(mixed_periodicity)
    numbers, positions, lattice, periodic, shifts = batch
    energy = energy_fn(shifts)

    ref = torch.stack(per_cell(energy, batch))
    out = torch.func.vmap(energy)(numbers, positions, lattice, periodic)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()

    # and the same as the batched call without any transform
    batched = dftd3(
        numbers,
        positions,
        param,
        lattice=lattice,
        periodic=periodic,
        cutoff=cutoff,
    ).sum(-1)
    assert pytest.approx(batched.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("mixed_periodicity", [False, True])
@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_vmap_jac(mixed_periodicity: bool, jac: str) -> None:
    """Gradients with respect to positions and lattice under `vmap`."""
    batch = setup(mixed_periodicity)
    numbers, positions, lattice, periodic, shifts = batch

    grad = getattr(torch.func, jac)(energy_fn(shifts), argnums=(1, 2))
    out_pos, out_lat = torch.func.vmap(grad)(
        numbers, positions, lattice, periodic
    )

    for i, (ref_pos, ref_lat) in enumerate(autograd_gradients(batch)):
        assert pytest.approx(ref_pos.cpu(), abs=tol) == out_pos[i].cpu()
        assert pytest.approx(ref_lat.cpu(), abs=tol) == out_lat[i].cpu()

    # nothing reaches the padding atoms
    assert (out_pos[numbers == 0] == 0).all()


@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_jac_single(jac: str) -> None:
    """Forward and reverse mode agree on one cell, without `vmap`."""
    batch = setup()
    numbers, positions, lattice, periodic, shifts = batch

    grad = getattr(torch.func, jac)(energy_fn(shifts), argnums=(1, 2))
    out_pos, out_lat = grad(numbers[0], positions[0], lattice[0], periodic[0])

    ref_pos, ref_lat = autograd_gradients(batch)[0]
    assert pytest.approx(ref_pos.cpu(), abs=tol) == out_pos.cpu()
    assert pytest.approx(ref_lat.cpu(), abs=tol) == out_lat.cpu()


@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_jac_of_vmap(jac: str) -> None:
    """`jac(vmap(...))` over the whole batch (outer derivative)."""
    batch = setup()
    numbers, positions, lattice, periodic, shifts = batch
    energy = energy_fn(shifts)

    def total(p: Tensor, lat: Tensor) -> Tensor:
        return torch.func.vmap(energy)(numbers, p, lat, periodic).sum()

    out_pos, out_lat = getattr(torch.func, jac)(total, argnums=(0, 1))(
        positions, lattice
    )

    for i, (ref_pos, ref_lat) in enumerate(autograd_gradients(batch)):
        assert pytest.approx(ref_pos.cpu(), abs=tol) == out_pos[i].cpu()
        assert pytest.approx(ref_lat.cpu(), abs=tol) == out_lat[i].cpu()


@pytest.mark.parametrize("outer", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("inner", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("argnum", [1, 2])
def test_vmap_hessian(outer: str, inner: str, argnum: int) -> None:
    """
    Second derivatives with respect to the positions (1) or the lattice
    (2), under `vmap`, for all forward/reverse mixes.
    """
    batch = setup(mixed_periodicity=True)
    numbers, positions, lattice, periodic, shifts = batch
    energy = energy_fn(shifts)

    def hess(a: str, b: str) -> Callable[..., Tensor]:
        first = getattr(torch.func, b)(energy, argnums=argnum)
        return getattr(torch.func, a)(first, argnums=argnum)

    # `jacrev(jacrev)` of each cell alone is the reference
    ref = torch.stack(per_cell(hess("jacrev", "jacrev"), batch))
    out = torch.func.vmap(hess(outer, inner))(
        numbers, positions, lattice, periodic
    )

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("name", ["rcov", "r4r2"])
def test_vmap_jac_table(jac: str, name: str) -> None:
    """Per-element gradients of a padded batch of cells, shared table."""
    batch = setup()
    numbers, positions, lattice, periodic, shifts = batch
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    table = {"rcov": radii.COV_D3, "r4r2": data.R4R2}[name](**dd)

    def energy(n: Tensor, p: Tensor, lat: Tensor, t: Tensor) -> Tensor:
        return dftd3(
            n,
            p,
            param,
            lattice=lat,
            shifts=shifts,
            cutoff=cutoff,
            **{f"{name}_table": t},
        ).sum()

    rev = torch.func.jacrev(energy, argnums=3)
    ref = torch.stack(
        [rev(numbers[i], positions[i], lattice[i], table) for i in range(2)]
    )

    grad = getattr(torch.func, jac)(energy, argnums=3)
    out = torch.func.vmap(grad, in_dims=(0, 0, 0, None))(
        numbers, positions, lattice, table
    )

    assert torch.isfinite(out).all()
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
    assert (out[:, 0] == 0).all()


def test_vmap_without_shifts() -> None:
    """
    Over positions only, with a fixed lattice, the shifts need not be
    given: they are built from the lattice, which is not batched.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions0, lattice = random_cell(TRICLINIC, 4, dd, seed=22)
    _, positions1, _ = random_cell(TRICLINIC, 4, dd, seed=23)
    positions = torch.stack([positions0, positions1])

    def energy(p: Tensor) -> Tensor:
        return dftd3(numbers, p, param, lattice=lattice, cutoff=cutoff).sum()

    out = torch.func.vmap(torch.func.jacrev(energy))(positions)
    ref = torch.stack([torch.func.jacrev(energy)(p) for p in positions])

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
