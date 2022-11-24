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
Test model parameters.
"""

import pytest
import torch

from tad_dftd3 import model, reference, util

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gw_single(dtype):
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    ref = reference.Reference().type(dtype)
    cn = sample["cn"].type(dtype)
    refgw = sample["weights"].type(dtype)

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(weights) == refgw


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gw_batch(dtype):
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
    ref = reference.Reference().type(dtype)
    cn = util.pack(
        (
            sample1["cn"].type(dtype),
            sample2["cn"].type(dtype),
        )
    )
    refgw = util.pack(
        (
            sample1["weights"].type(dtype),
            sample2["weights"].type(dtype),
        )
    )

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(weights) == refgw


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_c6_single(dtype):
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    ref = reference.Reference().type(dtype)
    weights = sample["weights"].type(dtype)
    refc6 = sample["c6"].type(dtype)

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(c6) == refc6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_c6_batch(dtype):
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
    ref = reference.Reference().type(dtype)
    weights = util.pack(
        (
            sample1["weights"].type(dtype),
            sample2["weights"].type(dtype),
        )
    )
    refc6 = util.pack(
        (
            sample1["c6"].type(dtype),
            sample2["c6"].type(dtype),
        )
    )

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(c6) == refc6
