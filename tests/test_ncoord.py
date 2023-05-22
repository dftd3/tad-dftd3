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

from tad_dftd3 import data, ncoord, util

from .samples import samples


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        ncoord.coordination_number(numbers, positions, rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        ncoord.coordination_number(numbers, positions)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cn_single(dtype: torch.dtype) -> None:
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    cutoff = torch.tensor(25, dtype=dtype)
    ref = sample["cn"].type(dtype)

    rcov = data.covalent_rad_d3[numbers]
    cn = ncoord.coordination_number(
        numbers, positions, rcov, ncoord.exp_count, cutoff=cutoff
    )
    assert cn.dtype == dtype
    assert pytest.approx(cn) == ref


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cn_batch(dtype: torch.dtype) -> None:
    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = util.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = util.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    ref = util.pack(
        (
            sample1["cn"].type(dtype),
            sample2["cn"].type(dtype),
        )
    )

    cn = ncoord.coordination_number(numbers, positions)
    assert cn.dtype == dtype
    assert pytest.approx(cn) == ref
