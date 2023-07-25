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
from math import sqrt

import pytest
import torch

from tad_dftd3 import model, reference, util

from .samples import samples
from .utils import get_device_from_str

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_gw_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"]
    ref = reference.Reference(dtype=dtype)
    cn = sample["cn"].type(dtype)
    refgw = sample["weights"].type(dtype)

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(refgw, abs=tol, rel=tol) == weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_gw_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = (
        samples[name1],
        samples[name2],
    )
    numbers = util.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    ref = reference.Reference(dtype=dtype)
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
    assert pytest.approx(refgw, abs=tol, rel=tol) == weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_c6_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"]
    ref = reference.Reference(dtype=dtype)
    weights = sample["weights"].type(dtype)
    refc6 = sample["c6"].type(dtype)

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(refc6, abs=tol, rel=tol) == c6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_c6_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = (
        samples[name1],
        samples[name2],
    )
    numbers = util.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    ref = reference.Reference(dtype=dtype)
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
    assert pytest.approx(refc6, abs=tol, rel=tol) == c6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_dtype(dtype: torch.dtype) -> None:
    ref = reference.Reference().type(dtype)
    assert ref.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_reference_device(device_str: str) -> None:
    device = get_device_from_str(device_str)
    ref = reference.Reference().to(device)
    assert ref.device == device

    with pytest.raises(AttributeError):
        ref.device = device


def test_reference_fail() -> None:
    c6 = reference._load_c6()  # pylint: disable=protected-access

    # wrong dtype
    with pytest.raises(RuntimeError):
        reference.Reference(c6=c6.type(torch.float16))

    # wrong device
    if torch.cuda.is_available() is True:
        with pytest.raises(RuntimeError):
            reference.Reference(c6=c6.to(torch.device("cuda")))

    # wrong shape
    with pytest.raises(RuntimeError):
        reference.Reference(
            cn=torch.rand((4, 4), dtype=torch.float32),
            c6=c6.type(torch.float32),
        )

    assert (
        repr(reference.Reference())
        == "Reference(n_element=95, n_reference=5, dtype=torch.float64, device=cpu)"
    )
