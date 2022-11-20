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

from tad_dftd3 import dftd3, util
from . import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_single(dtype):
    sample = samples.structures["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
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

    energy = dftd3(numbers, positions, param)

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

    energy = dftd3(numbers, positions, param)

    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)


@pytest.mark.grad
def test_param_grad():
    dtype = torch.float64
    sample = samples.structures["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    param = (
        torch.tensor(1.00000000, requires_grad=True, dtype=dtype),
        torch.tensor(0.78981345, requires_grad=True, dtype=dtype),
        torch.tensor(1.00000000, requires_grad=True, dtype=dtype),
        torch.tensor(0.49484001, requires_grad=True, dtype=dtype),
        torch.tensor(5.73083694, requires_grad=True, dtype=dtype),
    )
    label = ("s6", "s8", "s9", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: inputs[i] for i in range(len(inputs))}
        return dftd3(numbers, positions, input_param)

    assert torch.autograd.gradcheck(func, param)


@pytest.mark.grad
def test_positions_grad():
    dtype = torch.float64
    sample = samples.structures["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    positions.requires_grad_(True)
    param = {
        "s6": torch.tensor(1.00000000, dtype=dtype),
        "s8": torch.tensor(0.78981345, dtype=dtype),
        "s9": torch.tensor(1.00000000, dtype=dtype),
        "a1": torch.tensor(0.49484001, dtype=dtype),
        "a2": torch.tensor(5.73083694, dtype=dtype),
    }

    def func(positions):
        return dftd3(numbers, positions, param)

    assert torch.autograd.gradcheck(func, positions)
