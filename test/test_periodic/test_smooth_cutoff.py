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
"""
Smooth two-body cutoff for periodic cells, against s-dftd3: energy, gradient
and virial, dense and over a neighbour list, and the strain derivative by
central differences, which with a smooth cutoff needs no distance to stay
clear of it. The three-body term has no periodic evaluation.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import build_neighborlists
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import TRICLINIC, cells, param_on, random_cell
from ..conftest import DEVICE
from ..reference import reference_energy_per_atom, reference_gradient

# the images between 8 and 14 Bohr are switched off smoothly
cutoff = Cutoff(cn=15.0, disp2=14.0, width2=6.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("name", list(cells))
def test_energy_matches_reference(name: str, sparse: bool) -> None:
    structure = cells[name].to(DEVICE)
    par = param_on(DD64)

    ref = reference_energy_per_atom(structure, par, cutoff=cutoff)
    energy = dftd3(structure, par, cutoff=cutoff, sparse=sparse)

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("name", list(cells))
def test_gradient_and_virial_match_reference(name: str, sparse: bool) -> None:
    structure = cells[name].to(DEVICE)
    par = param_on(DD64)
    ref_grad, ref_virial = reference_gradient(structure, par, cutoff=cutoff)

    strain = torch.zeros(3, 3, requires_grad=True, **DD64)
    positions = structure.positions.clone().requires_grad_(True)
    deformed = structure.replace(
        positions=positions + positions @ strain,
        lattice=structure.lattice + structure.lattice @ strain,
    )
    energy = dftd3(deformed, par, cutoff=cutoff, sparse=sparse).sum()
    grad, virial = torch.autograd.grad(energy, (positions, strain))

    assert pytest.approx(ref_grad.cpu(), abs=tol, rel=0) == grad.cpu()
    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()


def test_batch_matches_reference() -> None:
    cells_ = [random_cell(TRICLINIC, n, DD64, seed=n) for n in (3, 5)]
    par = param_on(DD64)

    energy = dftd3(pack_structures(cells_), par, cutoff=cutoff)

    for i, cell in enumerate(cells_):
        ref = reference_energy_per_atom(cell, par, cutoff=cutoff)
        nat = ref.shape[-1]
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i, :nat].cpu()


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_virial_finite_difference_without_guard(name: str) -> None:
    """
    Central differences of the energy against the virial. With the hard
    cutoffs of `test_strain.py` this needs every pair to stay clear of the
    cutoff; with the smooth one, no pair crosses a discontinuity.
    """
    structure = cells[name].to(DEVICE)
    par = param_on(DD64)

    def energy(strain: Tensor) -> Tensor:
        deformed = structure.replace(
            positions=structure.positions + structure.positions @ strain,
            lattice=structure.lattice + structure.lattice @ strain,
        )
        return dftd3(deformed, par, cutoff=cutoff).sum()

    zero = torch.zeros(3, 3, requires_grad=True, **DD64)
    (virial,) = torch.autograd.grad(energy(zero), zero)

    step = 1e-5
    for i in range(3):
        for j in range(3):
            strain = torch.zeros(3, 3, **DD64)
            strain[i, j] = step
            numerical = (energy(strain) - energy(-strain)) / (2 * step)
            assert pytest.approx(virial[i, j].item(), abs=1e-10) == (
                numerical.item()
            )


def test_list_of_unstrained_cell_with_skin() -> None:
    structure = cells["urea"].to(DEVICE)
    par = param_on(DD64)
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2), skin=1.0
    )

    gen = torch.Generator().manual_seed(3)
    strain = 5e-3 * torch.randn(3, 3, generator=gen, dtype=torch.double)
    strain = strain.to(DEVICE)
    deformed = structure.replace(
        positions=structure.positions + structure.positions @ strain,
        lattice=structure.lattice + structure.lattice @ strain,
    )

    dense = dftd3(deformed, par, cutoff=cutoff)
    listed = dftd3(
        deformed, par, cutoff=cutoff, nbl_cn=nbl_cn, nbl_disp2=nbl_disp2
    )
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == listed.cpu()
