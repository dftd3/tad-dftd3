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
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float32])
def test_disp_atm(dtype):
    sample = samples["SiH4"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["energy"].type(dtype)
    param = {
        "s6": 1.0,
        "s8": 1.2576,
        "s9": 1.0,
        "alp": 14.0,
        "a1": 0.3768,
        "a2": 4.5865,
    }

    energy = dftd3(numbers, positions, param)

    assert energy.dtype == dtype
    # assert pytest.approx(energy) == ref
