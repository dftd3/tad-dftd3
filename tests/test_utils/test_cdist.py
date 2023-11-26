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
Test the utility functions.
"""

import pytest
import torch

from tad_dftd3 import utils
from tad_dftd3._typing import DD

from ..conftest import DEVICE as device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-6 if dtype == torch.float else 1e-14

    x = torch.randn(2, 3, 4, **dd)

    d1 = utils.cdist(x)
    d2 = utils.distance.cdist_direct_expansion(x, x, p=2)
    d3 = utils.distance.euclidean_dist_quadratic_expansion(x, x)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()
    assert pytest.approx(d2.cpu(), abs=tol) == d3.cpu()
    assert pytest.approx(d3.cpu(), abs=tol) == d1.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [2, 3, 4, 5])
def test_ps(dtype: torch.dtype, p: int) -> None:
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-6 if dtype == torch.float else 1e-14

    x = torch.randn(2, 4, 5, **dd)
    y = torch.randn(2, 4, 5, **dd)

    d1 = utils.cdist(x, y, p=p)
    d2 = torch.cdist(x, y, p=p)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()
