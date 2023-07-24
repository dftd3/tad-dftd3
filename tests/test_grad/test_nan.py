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
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd3 import dftd3

tol = 1e-8

device = None


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_grad_nan(dtype: torch.dtype) -> None:
    dd = {"device": device, "dtype": dtype}

    numbers = torch.tensor(
        [6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 7, 8, 8, 8], device=device
    )
    positions = torch.tensor(
        [
            [-1.0981, +0.1496, +0.1346],
            [-0.4155, +1.2768, +0.3967],
            [+0.9426, +0.7848, +0.1307],
            [+2.1708, +1.3814, -0.0347],
            [+3.3234, +0.5924, -0.1535],
            [+3.1564, -0.8110, -0.0285],
            [+1.8929, -1.4673, +0.0373],
            [+0.8498, -0.5613, +0.0109],
            [-0.7751, +2.2970, +0.5540],
            [+2.3079, +2.4725, -0.1905],
            [+4.3031, +0.9815, -0.4599],
            [+4.0011, -1.4666, -0.0514],
            [+1.8340, -2.5476, -0.1587],
            [-2.5629, -0.0306, -0.1458],
            [-3.0792, +1.0280, -0.3225],
            [-3.0526, -1.1594, +0.1038],
            [-0.4839, -0.9612, -0.0048],
        ],
        **dd,
    )

    param = {
        "s6": positions.new_tensor(1.00000000),
        "s8": positions.new_tensor(0.78981345),
        "s9": positions.new_tensor(1.00000000),
        "a1": positions.new_tensor(0.49484001),
        "a2": positions.new_tensor(5.73083694),
    }

    positions.requires_grad_(True)

    energy = dftd3(numbers, positions, param)
    assert not torch.isnan(energy).any(), "Energy contains NaN values"

    energy.sum().backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert not torch.isnan(grad_backward).any(), "Gradient contains NaN values"
