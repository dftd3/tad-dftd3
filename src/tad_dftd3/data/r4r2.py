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
Data: Expectation values
========================

PBE0/def2-QZVP atomic values calculated by S. Grimme in Gaussian (2010),
rare gases recalculated by J. Mewes with PBE0/aug-cc-pVQZ in Dirac (2018).
Also new super heavies Cn, Nh, Fl, Lv, Og and Am-Rg calculated at
4c-PBE/Dyall-AE4Z (Dirac 2022).
"""
from __future__ import annotations

import torch

__all__ = ["R4R2"]


def R4R2(
    dtype: torch.dtype | None = None, device: torch.device | None = None
) -> torch.Tensor:
    """
    Returns the r⁴ over r² expectation values as a tensor.

    Parameters
    ----------
    dtype : torch.dtype | None, optional
        The desired data type of the returned tensor. Defaults to None.
    device : torch.device | None, optional
        The desired device of the returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        A tensor containing the r⁴ over r² expectation values.
    """

    # Actually calculated r⁴ over r² expectation values
    # fmt: off
    _r4_over_r2 = [
        0.0000,  # None
        8.0589, 3.4698,  # H,He
        29.0974,14.8517,11.8799, 7.8715, 5.5588, 4.7566, 3.8025, 3.1036,  # Li-Ne
        26.1552,17.2304,17.7210,12.7442, 9.5361, 8.1652, 6.7463, 5.6004,  # Na-Ar
        29.2012,22.3934,  # K,Ca
        19.0598,16.8590,15.4023,12.5589,13.4788,  # Sc-
        12.2309,11.2809,10.5569,10.1428, 9.4907,  # -Zn
        13.4606,10.8544, 8.9386, 8.1350, 7.1251, 6.1971,  # Ga-Kr
        30.0162,24.4103,  # Rb,Sr
        20.3537,17.4780,13.5528,11.8451,11.0355,  # Y-
        10.1997, 9.5414, 9.0061, 8.6417, 8.9975,  # -Cd
        14.0834,11.8333,10.0179, 9.3844, 8.4110, 7.5152,  # In-Xe
        32.7622,27.5708,  # Cs,Ba
        23.1671,21.6003,20.9615,20.4562,20.1010,19.7475,19.4828,  # La-Eu
        15.6013,19.2362,17.4717,17.8321,17.4237,17.1954,17.1631,  # Gd-Yb
        14.5716,15.8758,13.8989,12.4834,11.4421,  # Lu-
        10.2671, 8.3549, 7.8496, 7.3278, 7.4820,  # -Hg
        13.5124,11.6554,10.0959, 9.7340, 8.8584, 8.0125,  # Tl-Rn
        29.8135,26.3157,  # Fr,Ra
        19.1885,15.8542,16.1305,15.6161,15.1226,16.1576,14.6510,  # Ac-Am
        14.7178,13.9108,13.5623,13.2326,12.9189,12.6133,12.3142,  # Cm-No
        14.8326,12.3771,10.6378, 9.3638, 8.2297,  # Lr-
        7.5667, 6.9456, 6.3946, 5.9159, 5.4929,  # -Cn
        6.7286, 6.5144,10.9169,10.3600, 9.4723, 8.6641,  # Nh-Og
    ]
    # fmt: on

    sqrtz = torch.sqrt(
        torch.arange(len(_r4_over_r2), device=device, dtype=dtype)
    )
    return torch.sqrt(
        0.5 * (torch.tensor(_r4_over_r2, device=device, dtype=dtype) * sqrtz)
    )
