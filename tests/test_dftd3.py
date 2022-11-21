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
from __future__ import annotations

import pytest
import torch

from tad_dftd3 import dftd3, util

from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_single(dtype):
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["disp2"].type(dtype)

    param = {
        "a1": positions.new_tensor(0.49484001),
        "s8": positions.new_tensor(0.78981345),
        "a2": positions.new_tensor(5.73083694),
    }

    energy = dftd3(numbers, positions, param)

    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp_batch(dtype):
    sample1, sample2 = (samples["PbH4-BiH3"], samples["C6H5I-CH3SH"])
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
            sample1["disp2"].type(dtype),
            sample2["disp2"].type(dtype),
        )
    )

    param = {
        "a1": positions.new_tensor(0.49484001),
        "s8": positions.new_tensor(0.78981345),
        "a2": positions.new_tensor(5.73083694),
    }

    energy = dftd3(numbers, positions, param)

    assert energy.dtype == dtype
    assert torch.allclose(energy, ref)


@pytest.mark.grad
def test_param_grad():
    dtype = torch.float64
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    param = (
        positions.new_tensor(1.00000000).requires_grad_(True),
        positions.new_tensor(0.78981345).requires_grad_(True),
        positions.new_tensor(1.00000000).requires_grad_(True),
        positions.new_tensor(0.49484001).requires_grad_(True),
        positions.new_tensor(5.73083694).requires_grad_(True),
    )
    label = ("s6", "s8", "s9", "a1", "a2")

    def func(*inputs):
        input_param = {label[i]: inputs[i] for i in range(len(inputs))}
        return dftd3(numbers, positions, input_param)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, param)


@pytest.mark.grad
def test_positions_grad():
    dtype = torch.float64
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    param = {
        "s6": positions.new_tensor(1.00000000),
        "s8": positions.new_tensor(0.78981345),
        "s9": positions.new_tensor(1.00000000),
        "a1": positions.new_tensor(0.49484001),
        "a2": positions.new_tensor(5.73083694),
    }

    pos = positions.detach().clone().requires_grad_(True)

    def func(positions):
        return dftd3(numbers, positions, param)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, pos)
