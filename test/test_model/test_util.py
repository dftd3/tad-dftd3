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
Test model utility.
"""
from unittest.mock import patch

import pytest
import torch

from tad_dftd3.model.c6 import _check_memory

from ..conftest import DEVICE

tol = 1e-8


@patch("tad_mctc.tools.memory.memory_device")
def test_memory_total(mock_memory) -> None:
    mock_memory.return_value = (0, 0)

    x = torch.randn((1000000,), device=DEVICE, dtype=torch.double)
    with pytest.raises(MemoryError):
        _check_memory(x, x)


@patch("tad_mctc.tools.memory.memory_device")
def test_memory_free(mock_memory) -> None:
    mock_memory.return_value = (0, 1e10)

    x = torch.randn((10000,), device=DEVICE, dtype=torch.double)
    with pytest.warns(ResourceWarning):
        _check_memory(x, x)
