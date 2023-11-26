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
Test coordination number.
"""
import pytest
import torch

from tad_dftd3 import data, ncoord, utils
from tad_dftd3._typing import DD

from ..conftest import DEVICE as device
from ..samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    cutoff = torch.tensor(25, **dd)
    ref = sample["cn"].to(**dd)

    rcov = data.covalent_rad_d3.to(**dd)[numbers]
    cn = ncoord.coordination_number(
        numbers, positions, ncoord.exp_count, rcov, cutoff=cutoff
    )
    assert cn.dtype == dtype
    assert pytest.approx(cn.cpu()) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = utils.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = utils.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = utils.pack(
        (
            sample1["cn"].to(**dd),
            sample2["cn"].to(**dd),
        )
    )

    cn = ncoord.coordination_number(numbers, positions)
    assert cn.dtype == dtype
    assert pytest.approx(cn.cpu()) == ref.cpu()
