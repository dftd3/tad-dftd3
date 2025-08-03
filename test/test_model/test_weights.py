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
Test the weights.
"""
import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.typing import DD

from tad_dftd3 import model, reference

from ..conftest import DEVICE
from .samples import samples

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)
    cn = sample["cn"].to(**dd)
    refgw = sample["weights"].to(**dd)

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(refgw.cpu(), abs=tol, rel=tol) == weights.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample1, sample2 = (
        samples[name1],
        samples[name2],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    ref = reference.Reference(**dd)
    cn = pack(
        (
            sample1["cn"].to(**dd),
            sample2["cn"].to(**dd),
        )
    )
    refgw = pack(
        (
            sample1["weights"].to(**dd),
            sample2["weights"].to(**dd),
        )
    )

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(refgw.cpu(), abs=tol, rel=tol) == weights.cpu()
