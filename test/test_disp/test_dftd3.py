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
Test calculation of dispersion energy and nuclear gradients.
"""

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.data.molecules import mols as samples
from tad_mctc.typing import DD

from tad_dftd3 import damping, data, dftd3, model, reference
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.ncoord import exp_count

from ..conftest import DEVICE
from ..reference import reference_energy_per_atom


def test_fail() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": torch.tensor(1.0000),
        "s8": torch.tensor(1.2576),
        "s9": torch.tensor(1.0000),
        "alp": torch.tensor(14.00),
        "a1": torch.tensor(0.3768),
        "a2": torch.tensor(4.5865),
    }

    # unsupported element
    with pytest.raises(ValueError):
        dftd3(torch.tensor([1, 105]), positions, param)

    # wrong numbers
    with pytest.raises(ValueError):
        dftd3(torch.tensor([1]), positions, param)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["LiH", "SiH4", "PbH4-BiH3"])
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(1.2576, **dd),
        "s9": torch.tensor(1.0000, **dd),
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.3768, **dd),
        "a2": torch.tensor(4.5865, **dd),
    }
    # No cutoff for the reference, so it uses s-dftd3's own; the explicit
    # `Cutoff()` below has to reproduce those.
    ref = reference_energy_per_atom(numbers, positions, param)

    rcov = radii.COV_D3(**dd)[numbers]
    rvdw = radii.VDW_PAIRWISE(**dd)[
        numbers.unsqueeze(-1), numbers.unsqueeze(-2)
    ]
    r4r2 = data.R4R2(**dd)[numbers]

    energy = dftd3(
        numbers,
        positions,
        param,
        ref=reference.Reference(**dd),
        rcov=rcov,
        rvdw=rvdw,
        r4r2=r4r2,
        cutoff=Cutoff(**dd),
        counting_function=exp_count,
        weighting_function=model.gaussian_weight,
        damping_function=damping.rational_damping,
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = (samples["PbH4-BiH3"], samples["C6H5I-CH3SH"])
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(1.2576, **dd),
        "s9": torch.tensor(0.0000, **dd),  # no ATM!
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.3768, **dd),
        "a2": torch.tensor(4.5865, **dd),
    }

    # s-dftd3 has no notion of a batch of independent molecules; compute the
    # reference for each molecule on its own and pack them the same way the
    # inputs above were packed.
    ref = pack(
        (
            reference_energy_per_atom(
                sample1["numbers"].to(DEVICE),
                sample1["positions"].to(**dd),
                param,
            ),
            reference_energy_per_atom(
                sample2["numbers"].to(DEVICE),
                sample2["positions"].to(**dd),
                param,
            ),
        )
    )

    energy = dftd3(numbers, positions, param)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()
