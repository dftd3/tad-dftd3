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
Smooth real-space cutoff of the two- and three-body terms.

With a width ``w > 0`` the contribution of a distance between ``cutoff - w``
and ``cutoff`` is scaled by a quintic that goes from 1 to 0, as in s-dftd3
(``width2``, ``width3`` of its ``realspace_cutoff``), instead of being cut
abruptly. The energies, gradients and Hessians are compared to s-dftd3 with
the cutoffs chosen so that many pairs and triples lie in the switching
region.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import build_neighborlist, build_neighborlists
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff, smooth_cutoff

from ..conftest import DEVICE
from ..reference import (
    reference_energy_per_atom,
    reference_gradient,
    reference_hessian,
    reference_pairwise,
)
from ..utils import load_structure

DD64: DD = {"device": DEVICE, "dtype": torch.double}

# TPSS0-D3(BJ)-ATM
param = {
    "s6": torch.tensor(1.0000, **DD64),
    "s8": torch.tensor(1.2576, **DD64),
    "s9": torch.tensor(1.0000, **DD64),
    "alp": torch.tensor(14.00, **DD64),
    "a1": torch.tensor(0.3768, **DD64),
    "a2": torch.tensor(4.5865, **DD64),
}

# short cutoffs, a wide switching region: most pairs of the molecules below
# lie inside it
cutoff = Cutoff(cn=15.0, disp2=12.0, disp3=9.0, width2=6.0, width3=4.0)

molecules = [("heavy28", "pbh4_bih3"), ("other", "C6H5I-CH3SH")]

tol = 1e-12


########################################################################
# The switch itself


def test_values() -> None:
    r = torch.tensor([0.0, 5.9, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], **DD64)
    sw = smooth_cutoff(r, cutoff=10.0, width=4.0)

    # 1 up to `cutoff - width`, 0 from `cutoff`, 1/2 in the middle
    assert sw[:3].tolist() == [1.0, 1.0, 1.0]
    assert sw[-2:].tolist() == [0.0, 0.0]
    assert pytest.approx(sw[4].item(), abs=1e-15) == 0.5

    # x = 3/4 and 1/4: x^3 (10 - 15 x + 6 x^2)
    assert pytest.approx(sw[3].item(), abs=1e-15) == 0.896484375
    assert pytest.approx(sw[5].item(), abs=1e-15) == 0.103515625


@pytest.mark.parametrize("width", [0.0, 10.0, 20.0])
def test_hard_cutoff(width: float) -> None:
    """A width of zero, or of at least the cutoff, is no switch at all."""
    r = torch.linspace(0.1, 15.0, 20, **DD64)
    assert (smooth_cutoff(r, cutoff=10.0, width=width) == 1.0).all()


def test_monotonic_and_smooth() -> None:
    r = torch.linspace(5.0, 10.0, 2001, **DD64)
    sw = smooth_cutoff(r, cutoff=10.0, width=5.0)

    assert (sw[1:] <= sw[:-1]).all()

    # first and second derivatives vanish at both edges (C2)
    r = torch.tensor([5.0, 10.0], requires_grad=True, **DD64)
    (d1,) = torch.autograd.grad(
        smooth_cutoff(r, 10.0, 5.0).sum(), r, create_graph=True
    )
    (d2,) = torch.autograd.grad(d1.sum(), r)
    assert pytest.approx(d1.tolist(), abs=1e-14) == [0.0, 0.0]
    assert pytest.approx(d2.tolist(), abs=1e-14) == [0.0, 0.0]


@pytest.mark.grad
def test_gradcheck() -> None:
    r = torch.tensor([5.5, 7.0, 8.5, 9.9], requires_grad=True, **DD64)
    assert dgradcheck(lambda x: smooth_cutoff(x, 10.0, 5.0), r)


def test_cutoff_fields() -> None:
    assert (Cutoff().width2, Cutoff().width3) == (0.0, 0.0)
    assert Cutoff(width2=3).width2 == 3.0

    with pytest.raises(ValueError, match="width2"):
        Cutoff(width2=-1.0)
    with pytest.raises(TypeError, match="width3"):
        Cutoff(width3=torch.tensor(1.0))  # type: ignore[arg-type]


########################################################################
# Against s-dftd3: molecules


@pytest.mark.parametrize("source", molecules)
def test_energy_matches_reference(source: tuple[str, str]) -> None:
    structure = load_structure(*source, DD64)

    ref = reference_energy_per_atom(structure, param, cutoff=cutoff)
    energy = dftd3(structure, param, cutoff=cutoff)

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()

    # only meaningful if the switch changes the energy
    hard = Cutoff(cn=15.0, disp2=12.0, disp3=9.0)
    assert not torch.allclose(
        energy, dftd3(structure, param, cutoff=hard), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("source", molecules)
def test_terms_match_reference(source: tuple[str, str]) -> None:
    """The two-body and three-body energies alone, pair by pair."""
    structure = load_structure(*source, DD64)

    for field, s9 in (("width2", 0.0), ("width3", 1.0)):
        cut = Cutoff(
            **{
                **cutoff.__dict__,
                **{
                    "width2": cutoff.width2 if field == "width2" else 0.0,
                    "width3": cutoff.width3 if field == "width3" else 0.0,
                },
            }
        )
        par = {**param, "s9": torch.tensor(s9, **DD64)}

        two, three = reference_pairwise(structure, par, cutoff=cut)
        ref = two.sum(-1) + three.sum(-1)
        energy = dftd3(structure, par, cutoff=cut)

        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


@pytest.mark.parametrize("source", molecules)
def test_gradient_matches_reference(source: tuple[str, str]) -> None:
    structure = load_structure(*source, DD64)
    ref, _ = reference_gradient(structure, param, cutoff=cutoff)

    positions = structure.positions.clone().requires_grad_(True)
    energy = dftd3(
        structure.replace(positions=positions), param, cutoff=cutoff
    ).sum()
    (grad,) = torch.autograd.grad(energy, positions)

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == grad.cpu()


def test_hessian_matches_reference() -> None:
    structure = load_structure("heavy28", "pbh4_bih3", DD64)
    ref = reference_hessian(structure, param, cutoff=cutoff)

    def energy(positions: Tensor) -> Tensor:
        return dftd3(
            structure.replace(positions=positions), param, cutoff=cutoff
        ).sum()

    hess = torch.func.hessian(energy)(structure.positions)

    assert pytest.approx(ref.cpu(), abs=1e-12, rel=0) == hess.cpu()


def test_batch_matches_reference() -> None:
    structures = [load_structure(*src, DD64) for src in molecules]

    energy = dftd3(pack_structures(structures), param, cutoff=cutoff)

    for i, structure in enumerate(structures):
        ref = reference_energy_per_atom(structure, param, cutoff=cutoff)
        nat = ref.shape[-1]
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i, :nat].cpu()


########################################################################
# Sparse against dense


@pytest.mark.parametrize("source", molecules)
def test_sparse_matches_dense(source: tuple[str, str]) -> None:
    structure = load_structure(*source, DD64)
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2)
    )
    nbl_disp3 = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, param, cutoff=cutoff)
    sparse = dftd3(
        structure,
        param,
        cutoff=cutoff,
        nbl_cn=nbl_cn,
        nbl_disp2=nbl_disp2,
        nbl_disp3=nbl_disp3,
    )

    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == sparse.cpu()
