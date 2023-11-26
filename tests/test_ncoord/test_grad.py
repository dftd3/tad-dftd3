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
Test derivative (w.r.t. positions) of the exponential and error counting
functions used for the coordination number.
"""
import pytest
import torch

from tad_dftd3._typing import DD, CountingFunction
from tad_dftd3.ncoord import dexp_count, exp_count

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("function", [(exp_count, dexp_count)])
def test_count_grad(
    dtype: torch.dtype, function: tuple[CountingFunction, CountingFunction]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    tol = torch.finfo(dtype).eps ** 0.5 * 10
    cf, dcf = function

    a = torch.rand(4, **dd)
    b = torch.rand(4, **dd)

    a_grad = a.detach().clone().requires_grad_(True)
    count = cf(a_grad, b)

    grad_auto = torch.autograd.grad(count.sum(-1), a_grad)[0]
    grad_expl = dcf(a, b)

    assert pytest.approx(grad_auto.cpu(), abs=tol) == grad_expl.cpu()
