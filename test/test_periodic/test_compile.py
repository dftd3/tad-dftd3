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
Testing `torch.compile(fullgraph=True)` of the dense periodic energy, with
the periodic image shifts built beforehand.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc._version import __tversion__
from tad_mctc.batch import pack
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import cells, random_cell
from ..conftest import DEVICE, compile_test, requires_compile


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Isolate compile state between tests."""
    torch._dynamo.reset()  # pylint: disable=protected-access
    yield
    torch._dynamo.reset()  # pylint: disable=protected-access


tol = 1e-10

param = {
    "s6": torch.tensor(1.0000, dtype=torch.double),
    "s8": torch.tensor(1.2576, dtype=torch.double),
    "a1": torch.tensor(0.3768, dtype=torch.double),
    "a2": torch.tensor(4.5865, dtype=torch.double),
}

cutoff = Cutoff(cn=10.0, disp2=12.0)


def _setup(name: str) -> tuple[Tensor, Tensor, Tensor, PeriodicShifts]:
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])
    periodic = torch.ones(3, dtype=torch.bool, device=DEVICE)
    shifts = build_periodic_shifts(
        lattice, periodic, max(cutoff.cn, cutoff.disp2)
    )
    return numbers, positions, lattice, shifts


@requires_compile
@pytest.mark.parametrize("s9", [None, 0.0])
def test_fullgraph(s9: float | None) -> None:
    """
    A single graph for a cell, from a cold start, with `s9` left out or
    given as the Python number 0.0.
    """
    numbers, positions, lattice, shifts = _setup("periodic_triclinic")
    par = {**param} if s9 is None else {**param, "s9": s9}

    def energy(n: Tensor, p: Tensor, lat: Tensor) -> Tensor:
        return dftd3(n, p, par, lattice=lat, shifts=shifts, cutoff=cutoff)

    ref = energy(numbers, positions, lattice)
    out = compile_test(energy, fullgraph=True)(numbers, positions, lattice)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@requires_compile
def test_fullgraph_batch() -> None:
    """A padded batch of cells of different shape and periodicity."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers0, positions0, lattice0 = (t.to(DEVICE) for t in cells["urea"])
    numbers1, positions1, lattice1 = random_cell(
        6.0 * torch.eye(3, dtype=torch.double), 4, dd, seed=31
    )

    numbers = pack([numbers0, numbers1])
    positions = pack([positions0, positions1])
    lattice = torch.stack([lattice0, lattice1])
    periodic = torch.tensor([[True, True, True], [True, False, True]])
    shifts = build_periodic_shifts(
        lattice, periodic, max(cutoff.cn, cutoff.disp2)
    )

    def energy(p: Tensor, lat: Tensor) -> Tensor:
        return dftd3(
            numbers,
            p,
            param,
            lattice=lat,
            periodic=periodic,
            shifts=shifts,
            cutoff=cutoff,
        )

    ref = energy(positions, lattice)
    out = compile_test(energy, fullgraph=True)(positions, lattice)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@requires_compile
def test_fullgraph_autograd() -> None:
    """Gradients by `backward` through the compiled graph."""
    numbers, positions, lattice, shifts = _setup("urea")

    def energy(p: Tensor, lat: Tensor) -> Tensor:
        return dftd3(
            numbers, p, param, lattice=lat, shifts=shifts, cutoff=cutoff
        ).sum()

    def gradients(fn: Callable[..., Tensor]) -> tuple[Tensor, ...]:
        p = positions.clone().requires_grad_(True)
        lat = lattice.clone().requires_grad_(True)
        return torch.autograd.grad(fn(p, lat), (p, lat))

    ref = gradients(energy)
    out = gradients(compile_test(energy, fullgraph=True))

    for r, o in zip(ref, out):
        assert pytest.approx(r.cpu(), abs=tol) == o.cpu()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 5, 0),
    reason="`torch.compile` of `torch.func` transforms needs PyTorch 2.5.0.",
)
@pytest.mark.parametrize("transform", ["vmap(jacrev)", "jacfwd(vmap)"])
def test_fullgraph_vmap_jac(transform: str) -> None:
    """
    Compiled gradients with respect to positions and lattice, of a padded
    batch of cells under `vmap`.

    Forward mode is taken of the `vmap`, not under it, and with respect to
    one argument at a time: PyTorch (2.10) cannot compile `jacfwd` under
    `vmap` with an integer first argument, nor `jacfwd` with several
    `argnums`, even for functions as simple as ``(x**2 * n).sum()`` and
    ``(a @ b).sum()``.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers0, positions0, lattice0 = (
        t.to(DEVICE) for t in cells["periodic_triclinic"]
    )
    numbers1, positions1, lattice1 = random_cell(
        6.0 * torch.eye(3, dtype=torch.double), 3, dd, seed=32
    )

    numbers = pack([numbers0, numbers1])
    positions = pack([positions0, positions1])
    lattice = torch.stack([lattice0, lattice1])
    periodic = torch.ones(2, 3, dtype=torch.bool, device=DEVICE)
    shifts = build_periodic_shifts(
        lattice, periodic, max(cutoff.cn, cutoff.disp2)
    )

    def energy(n: Tensor, p: Tensor, lat: Tensor) -> Tensor:
        return dftd3(
            n, p, param, lattice=lat, shifts=shifts, cutoff=cutoff
        ).sum()

    def total(p: Tensor, lat: Tensor) -> Tensor:
        return torch.func.vmap(energy)(numbers, p, lat).sum()

    if transform == "vmap(jacrev)":
        jac = torch.func.vmap(torch.func.jacrev(energy, argnums=(1, 2)))
        grad = lambda p, lat: jac(numbers, p, lat)
    else:
        jac_pos = torch.func.jacfwd(total, argnums=0)
        jac_lat = torch.func.jacfwd(total, argnums=1)
        grad = lambda p, lat: (jac_pos(p, lat), jac_lat(p, lat))

    ref = grad(positions, lattice)
    out = compile_test(grad, fullgraph=True)(positions, lattice)

    for r, o in zip(ref, out):
        assert pytest.approx(r.cpu(), abs=tol) == o.cpu()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 5, 0),
    reason="`torch.compile` of `torch.func` transforms needs PyTorch 2.5.0.",
)
def test_fullgraph_hessian() -> None:
    """Compiled forward-over-reverse Hessian with respect to the lattice."""
    numbers, positions, lattice, shifts = _setup("periodic_triclinic")

    def energy(lat: Tensor) -> Tensor:
        return dftd3(
            numbers, positions, param, lattice=lat, shifts=shifts, cutoff=cutoff
        ).sum()

    hessian = torch.func.jacfwd(torch.func.jacrev(energy))
    ref = hessian(lattice)
    out = compile_test(hessian, fullgraph=True)(lattice)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
