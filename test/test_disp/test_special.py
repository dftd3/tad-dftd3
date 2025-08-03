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
Test calculation of dispersion energy for a system, which fail without the
weird handling of exceptional values in the calculation of the weights.
"""
import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data import radii

from tad_dftd3 import damping, data, dftd3, model, reference
from tad_dftd3.ncoord import exp_count
from tad_dftd3.typing import DD

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["La3N@C80"])
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["disp2"].to(**dd)

    rcov = radii.COV_D3(**dd)[numbers]
    rvdw = radii.VDW_PAIRWISE(**dd)[
        numbers.unsqueeze(-1), numbers.unsqueeze(-2)
    ]
    r4r2 = data.R4R2(**dd)[numbers]
    cutoff = torch.tensor(50, **dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(2.4000, **dd),
        "s9": torch.tensor(0.0000, **dd),
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.6300, **dd),
        "a2": torch.tensor(5.0000, **dd),
    }

    energy = dftd3(
        numbers,
        positions,
        param,
        ref=reference.Reference(**dd),
        rcov=rcov,
        rvdw=rvdw,
        r4r2=r4r2,
        cutoff=cutoff,
        counting_function=exp_count,
        weighting_function=model.gaussian_weight,
        damping_function=damping.rational_damping,
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = (samples["LiH"], samples["La3N@C80"])
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
    ref = pack(
        (
            torch.tensor(
                [
                    -4.1054019506089849e-05,
                    -4.1054019506089849e-05,
                ],
                **dd
            ),
            sample2["disp2"].to(**dd),
        )
    )

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(2.4000, **dd),
        "s9": torch.tensor(0.0000, **dd),
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.6300, **dd),
        "a2": torch.tensor(5.0000, **dd),
    }

    energy = dftd3(numbers, positions, param)
    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()
