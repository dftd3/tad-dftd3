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
Data: Radii
===========

Data arrays for atomic constants like covalent radii or van-der-Waals radii.

The `vdw_rad_d3` were previously stored explicitly in one list and then
reshaped to the required `(MAX_ELEMENT, MAX_ELEMENT)` tensor. For the old
version, see older commits (e.g. https://github.com/dftd3/tad-dftd3/blob/ecc50f19adb8aa8baa38a188d04228c4f26975d6/src/tad_dftd3/data/radii.py)
"""
import os.path as op
from typing import Optional

import torch
from tad_mctc.data.radii import COV_D3

from ..typing import Tensor

__all__ = ["COV_D3", "VDW_D3"]


def _load_vdw_rad_d3(
    dtype: torch.dtype = torch.double, device: Optional[torch.device] = None
) -> Tensor:
    """
    Load reference VDW radii from file.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Floating point precision for tensor. Defaults to `torch.double`.
    device : Optional[torch.device], optional
        Device of tensor. Defaults to None.

    Returns
    -------
    Tensor
        VDW radii.
    """
    path = op.join(op.dirname(__file__), "vdw-d3.pt")
    return torch.load(path).type(dtype).to(device)


VDW_D3 = _load_vdw_rad_d3()
