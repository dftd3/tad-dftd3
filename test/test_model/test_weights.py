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
from ..references import reference_cn, reference_weights
from ..utils import approx_ref, load_sample

sample_list: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),
    ("heavy28", "pbh4_bih3"),
    ("other", "C6H5I-CH3SH"),
    ("mb16_43", "01"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source", sample_list)
def test_single(dtype: torch.dtype, source: tuple[str, str]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = load_sample(*source, dd)[0]
    ref = reference.Reference(**dd)

    # coordination number and weights both come from s-dftd3's Fortran
    # library (see test/references)
    cn = reference_cn(*source, dd)
    refgw = reference_weights(*source, dd)

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert approx_ref(refgw.cpu(), dtype) == weights.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source1", [("mb16_43", "SiH4")])
@pytest.mark.parametrize("source2", sample_list)
def test_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        (
            load_sample(*source1, dd)[0],
            load_sample(*source2, dd)[0],
        )
    )
    ref = reference.Reference(**dd)
    # s-dftd3 has no notion of a batch of independent molecules; the
    # reference of each molecule is packed like the inputs above
    cn = pack((reference_cn(*source1, dd), reference_cn(*source2, dd)))
    refgw = pack(
        (reference_weights(*source1, dd), reference_weights(*source2, dd))
    )

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert approx_ref(refgw.cpu(), dtype) == weights.cpu()
