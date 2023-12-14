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
Test loading C6 coefficients.
"""
import torch

from tad_dftd3 import reference


def test_ref() -> None:
    c6_np = reference._load_c6_npy(dtype=torch.double)
    c6_pt = reference._load_c6_pt(dtype=torch.double)

    assert c6_np.shape == c6_pt.shape
    assert (c6_np == c6_pt).all()

    maxelem = 104  # 103 + dummy
    assert c6_np.shape == torch.Size((maxelem, maxelem, 7, 7))
