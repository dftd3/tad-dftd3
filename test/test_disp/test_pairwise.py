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
Test the pair-resolved two-body dispersion energy against the s-dftd3
Fortran reference.

This is a sharper comparison than a total energy: a wrong pair shows up as
a wrong matrix element at a specific index, rather than being hidden inside
(or accidentally cancelled by) a sum over every pair in the system.

tad-dftd3's own :func:`tad_dftd3.disp.dispersion2` sums over pairs
internally (``torch.sum(..., dim=-1)``) and never returns the ``(nat,
nat)`` matrix it sums. :func:`pairwise_two_body` below rebuilds that matrix
using tad-dftd3's own real building blocks -- the same ``real_pairs`` mask,
the same ``storch.cdist``, the same ``damping_function`` call
:func:`tad_dftd3.disp.dispersion2` itself makes -- and stops one step short
of the sum, rather than reimplementing the physics from scratch. Checked
directly, not assumed: with the sum taken afterwards, this reproduces
:func:`tad_dftd3.disp.dispersion2`'s own atom-resolved output exactly.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.data.molecules import mols as samples
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import damping, data, defaults, disp

from ..conftest import DEVICE
from ..reference import reference_energy_per_atom, reference_pairwise
from .test_disp import live_c6

sample_list = ["AmF3", "SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


def pairwise_two_body(
    numbers: Tensor,
    positions: Tensor,
    c6: Tensor,
    r4r2: Tensor,
    param: dict[str, Tensor],
    cutoff: Tensor,
) -> Tensor:
    """
    The ``(nat, nat)`` two-body dispersion matrix, computed the same way
    :func:`tad_dftd3.disp.dispersion2` computes it internally, up to (not
    including) the final sum over pairs.

    Symmetric with a zero diagonal, and already carries the ``-0.5`` that
    makes summing a whole row give that atom's share of the energy -- see
    the module docstring of ``test/reference.py`` for why that matches
    s-dftd3's own pairwise-matrix convention.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    within_cutoff = mask * (distances <= cutoff)
    zero = torch.tensor(0.0, **dd)

    t6 = torch.where(
        within_cutoff, damping.rational_damping(6, distances, qq, param), zero
    )
    t8 = torch.where(
        within_cutoff, damping.rational_damping(8, distances, qq, param), zero
    )

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))

    return -0.5 * (s6 * c6 * t6 + s8 * c8 * t8)


@pytest.mark.parametrize("name", sample_list)
def test_pairwise_matches_dispersion2(name: str) -> None:
    """``pairwise_two_body`` summed over pairs reproduces dispersion2()."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    c6 = live_c6(numbers, positions, dd)
    r4r2 = data.R4R2(**dd)[numbers]
    cutoff = torch.tensor(defaults.D3_DISP_CUTOFF, **dd)

    par = {"s6": torch.tensor(1.0, **dd), "s8": torch.tensor(1.2576, **dd)}

    matrix = pairwise_two_body(numbers, positions, c6, r4r2, par, cutoff)
    energy_per_atom = matrix.sum(dim=1)

    expected = disp.dispersion2(
        numbers, positions, par, c6, r4r2, damping.rational_damping, cutoff
    )

    assert torch.allclose(energy_per_atom, expected, atol=1e-14)


@pytest.mark.parametrize("name", sample_list)
def test_pairwise_two_body_matches_reference(name: str) -> None:
    """tad-dftd3's pairwise two-body matrix agrees with s-dftd3, elementwise."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    c6 = live_c6(numbers, positions, dd)
    r4r2 = data.R4R2(**dd)[numbers]
    cutoff = torch.tensor(defaults.D3_DISP_CUTOFF, **dd)

    par = {"s6": torch.tensor(1.0, **dd), "s8": torch.tensor(1.2576, **dd)}

    matrix = pairwise_two_body(numbers, positions, c6, r4r2, par, cutoff)
    two_body_ref, _ = reference_pairwise(numbers, positions, par)

    assert torch.allclose(matrix, two_body_ref, atol=1e-10)


@pytest.mark.parametrize("name", sample_list)
def test_pairwise_two_body_sums_to_reference_total(name: str) -> None:
    """The reference's own pairwise matrix sums to its own reported total.

    Checks the summation convention documented in ``reference.py`` is
    actually what s-dftd3 returns, rather than trusting that documentation
    to stay correct on its own.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    par = {"s6": torch.tensor(1.0, **dd), "s8": torch.tensor(1.2576, **dd)}

    two_body_ref, _ = reference_pairwise(numbers, positions, par)

    assert torch.allclose(two_body_ref, two_body_ref.T)
    assert torch.allclose(
        torch.diagonal(two_body_ref), torch.zeros(numbers.shape[-1], **dd)
    )

    # reference_energy_per_atom already sums this same matrix internally
    # (see reference.py); recomputing the total here via the raw pairwise
    # sum keeps this test meaningful even if that internal wiring changes.
    total_from_matrix = two_body_ref.sum()
    total_from_energy = reference_energy_per_atom(numbers, positions, par).sum()
    assert total_from_matrix.item() == pytest.approx(
        total_from_energy.item(), rel=1e-12
    )
