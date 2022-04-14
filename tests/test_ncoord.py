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

import pytest
import torch

from tad_dftd3 import data, ncoord, util
from . import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cn_single(dtype):
    sample = samples.structures["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    rcov = data.covalent_rad_d3[numbers]
    ref = torch.Tensor(
        [
            3.9388208389,
            0.9832025766,
            0.9832026958,
            0.9832026958,
            0.9865897894,
            2.9714603424,
            0.9870455265,
            0.9870456457,
            0.9870455265,
        ],
    ).type(dtype)

    cn = ncoord.coordination_number(positions, numbers, rcov, ncoord.exp_count)
    assert cn.dtype == dtype
    assert torch.allclose(cn, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cn_batch(dtype):
    sample1, sample2 = (
        samples.structures["PbH4-BiH3"],
        samples.structures["C6H5I-CH3SH"],
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
    rcov = data.covalent_rad_d3[numbers]
    ref = torch.Tensor(
        [
            [
                3.9388208389,
                0.9832025766,
                0.9832026958,
                0.9832026958,
                0.9865897894,
                2.9714603424,
                0.9870455265,
                0.9870456457,
                0.9870455265,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
            ],
            [
                3.1393690109,
                3.1313166618,
                3.1393768787,
                3.3153429031,
                3.1376547813,
                3.3148119450,
                1.5363609791,
                1.0035246611,
                1.0122337341,
                1.0036621094,
                1.0121959448,
                1.0036619902,
                2.1570565701,
                0.9981809855,
                3.9841127396,
                1.0146225691,
                1.0123561621,
                1.0085891485,
            ],
        ],
    ).type(dtype)

    cn = ncoord.coordination_number(positions, numbers, rcov, ncoord.exp_count)
    assert cn.dtype == dtype
    assert torch.allclose(cn, ref)
