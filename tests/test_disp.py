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

from tad_dftd3 import data, disp, util
from . import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_single(dtype):
    sample = samples.structures["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    c6 = sample["c6"].type(dtype)
    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = data.sqrt_z_r4_over_r2[numbers]
    param = dict(a1=0.49484001, s8=0.78981345, a2=5.73083694)
    ref = torch.tensor(
        [
            -3.5479912602e-04,
            -8.9124281989e-05,
            -8.9124287363e-05,
            -8.9124287363e-05,
            -1.3686794039e-04,
            -3.8805575850e-04,
            -8.7387460069e-05,
            -8.7387464149e-05,
            -8.7387460069e-05,
        ]
    ).type(dtype)

    energy = disp.dispersion(
        numbers, positions, param, c6, rvdw, r4r2, disp.rational_damping
    )

    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype):
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
    c6 = util.pack(
        (
            sample1["c6"].type(dtype),
            sample2["c6"].type(dtype),
        )
    )
    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = data.sqrt_z_r4_over_r2[numbers]
    param = dict(a1=0.49484001, s8=0.78981345, a2=5.73083694)
    ref = torch.tensor(
        [
            [
                -3.5479912602e-04,
                -8.9124281989e-05,
                -8.9124287363e-05,
                -8.9124287363e-05,
                -1.3686794039e-04,
                -3.8805575850e-04,
                -8.7387460069e-05,
                -8.7387464149e-05,
                -8.7387460069e-05,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
                -0.0000000000e-00,
            ],
            [
                -4.1551151549e-04,
                -3.9770287009e-04,
                -4.1552470565e-04,
                -4.4246829733e-04,
                -4.7527776799e-04,
                -4.4258484762e-04,
                -1.0637547378e-03,
                -1.5452322970e-04,
                -1.9695663808e-04,
                -1.6184434935e-04,
                -1.9703176496e-04,
                -1.6183339573e-04,
                -4.6648977616e-04,
                -1.3764556692e-04,
                -2.4555353368e-04,
                -1.3535967638e-04,
                -1.5719227870e-04,
                -1.1675684940e-04,
            ],
        ]
    ).type(dtype)

    energy = disp.dispersion(
        numbers, positions, param, c6, rvdw, r4r2, disp.rational_damping
    )

    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)
