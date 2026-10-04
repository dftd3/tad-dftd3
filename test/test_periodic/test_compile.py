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
from tad_mctc import Structure
from tad_mctc._version import __tversion__
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import cells, param, random_cell
from ..conftest import DEVICE, compile_test, requires_compile

pytestmark = pytest.mark.usefixtures("reset_dynamo")

requires_compiled_transforms = pytest.mark.skipif(
    __tversion__ < (2, 5, 0),
    reason="`torch.compile` of `torch.func` transforms needs PyTorch 2.5.0.",
)

requires_compiled_jacfwd = pytest.mark.skipif(
    __tversion__ < (2, 6, 0),
    reason="`torch.compile` of `jacfwd` needs PyTorch 2.6.0 (2.5 fails in "
    "`_jvp_treespec_compare` and on `push_jvp` arguments).",
)

tol = 1e-10

cutoff = Cutoff(cn=10.0, disp2=12.0)


def _shifts(structure: Structure) -> PeriodicShifts:
    """Shifts for both cutoffs, built beforehand as compilation needs."""
    assert structure.lattice is not None and structure.periodic is not None
    return build_periodic_shifts(
        structure.lattice, structure.periodic, max(cutoff.cn, cutoff.disp2)
    )


def _batch() -> Structure:
    """A padded batch of two cells of different shape and periodicity."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    return pack_structures(
        [
            cells["periodic_triclinic"].to(DEVICE),
            random_cell(
                6.0 * torch.eye(3, dtype=torch.double),
                3,
                dd,
                seed=32,
                periodic=torch.tensor([True, False, True], device=DEVICE),
            ),
        ]
    )


@requires_compile
@pytest.mark.parametrize("s9", [None, 0.0])
def test_fullgraph(s9: float | None) -> None:
    """
    A single graph for a cell, from a cold start, with `s9` left out or
    given as the Python number 0.0. The cell is an argument of the compiled
    function, as a `Structure`, also when one is replaced.
    """
    structure = cells["periodic_triclinic"].to(DEVICE)
    shifts = _shifts(structure)
    par = {**param} if s9 is None else {**param, "s9": s9}

    def energy(s: Structure) -> Tensor:
        return dftd3(s, par, shifts=shifts, cutoff=cutoff)

    compiled = compile_test(energy, fullgraph=True)
    moved = structure.replace(positions=structure.positions + 0.1)
    for s in (structure, moved):
        assert pytest.approx(energy(s).cpu(), abs=tol) == compiled(s).cpu()


@requires_compile
def test_fullgraph_batch() -> None:
    """A padded batch of cells of different shape and periodicity."""
    batch = _batch()
    shifts = _shifts(batch)

    def energy(s: Structure) -> Tensor:
        return dftd3(s, param, shifts=shifts, cutoff=cutoff)

    ref = energy(batch)
    out = compile_test(energy, fullgraph=True)(batch)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@requires_compile
def test_fullgraph_autograd() -> None:
    """Gradients by `backward` through the compiled graph."""
    structure = cells["urea"].to(DEVICE)
    shifts = _shifts(structure)
    assert structure.lattice is not None

    def energy(p: Tensor, lat: Tensor) -> Tensor:
        cell = structure.replace(positions=p, lattice=lat)
        return dftd3(cell, param, shifts=shifts, cutoff=cutoff).sum()

    def gradients(fn: Callable[..., Tensor]) -> tuple[Tensor, ...]:
        assert structure.lattice is not None
        p = structure.positions.clone().requires_grad_(True)
        lat = structure.lattice.clone().requires_grad_(True)
        return torch.autograd.grad(fn(p, lat), (p, lat))

    ref = gradients(energy)
    out = gradients(compile_test(energy, fullgraph=True))

    for r, o in zip(ref, out):
        assert pytest.approx(r.cpu(), abs=tol) == o.cpu()


@requires_compile
@requires_compiled_transforms
@pytest.mark.parametrize(
    "transform",
    [
        "vmap(jacrev)",
        pytest.param("jacfwd(vmap)", marks=requires_compiled_jacfwd),
    ],
)
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
    batch = _batch()
    shifts = _shifts(batch)
    numbers, periodic = batch.numbers, batch.periodic
    assert batch.lattice is not None and periodic is not None

    def energy(n: Tensor, p: Tensor, lat: Tensor, per: Tensor) -> Tensor:
        cell = Structure(numbers=n, positions=p, lattice=lat, periodic=per)
        return dftd3(cell, param, shifts=shifts, cutoff=cutoff).sum()

    def total(p: Tensor, lat: Tensor) -> Tensor:
        return torch.func.vmap(energy)(numbers, p, lat, periodic).sum()

    if transform == "vmap(jacrev)":
        jac = torch.func.vmap(torch.func.jacrev(energy, argnums=(1, 2)))
        grad = lambda p, lat: jac(numbers, p, lat, periodic)
    else:
        jac_pos = torch.func.jacfwd(total, argnums=0)
        jac_lat = torch.func.jacfwd(total, argnums=1)
        grad = lambda p, lat: (jac_pos(p, lat), jac_lat(p, lat))

    ref = grad(batch.positions, batch.lattice)
    out = compile_test(grad, fullgraph=True)(batch.positions, batch.lattice)

    for r, o in zip(ref, out):
        assert pytest.approx(r.cpu(), abs=tol) == o.cpu()


@requires_compile
@requires_compiled_transforms
@requires_compiled_jacfwd
def test_fullgraph_hessian() -> None:
    """Compiled forward-over-reverse Hessian with respect to the lattice."""
    structure = cells["periodic_triclinic"].to(DEVICE)
    shifts = _shifts(structure)

    def energy(lat: Tensor) -> Tensor:
        cell = structure.replace(lattice=lat)
        return dftd3(cell, param, shifts=shifts, cutoff=cutoff).sum()

    hessian = torch.func.jacfwd(torch.func.jacrev(energy))
    ref = hessian(structure.lattice)
    out = compile_test(hessian, fullgraph=True)(structure.lattice)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
