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
from tad_mctc.data.molecules import mols
from tad_mctc.typing import DD

from tad_dftd3 import model, reference

from ..conftest import DEVICE
from ..references import reference_cn, reference_weights

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    numbers = mols[name]["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)

    # cn and weights both come from s-dftd3's Fortran library, at
    # dftd3()'s own CN cutoff (see test/references). Feeding in the
    # reference CN isolates this test to weight_references, independent of
    # coordination_number.
    cn = reference_cn(name, dd)
    refgw = reference_weights(name, dd)

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(refgw.cpu(), abs=tol, rel=tol) == weights.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    numbers = pack(
        (
            mols[name1]["numbers"].to(DEVICE),
            mols[name2]["numbers"].to(DEVICE),
        )
    )
    ref = reference.Reference(**dd)

    cn = pack((reference_cn(name1, dd), reference_cn(name2, dd)))
    refgw = pack((reference_weights(name1, dd), reference_weights(name2, dd)))

    weights = model.weight_references(numbers, cn, ref, model.gaussian_weight)

    assert weights.dtype == dtype
    assert pytest.approx(refgw.cpu(), abs=tol, rel=tol) == weights.cpu()
