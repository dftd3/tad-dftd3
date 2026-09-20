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
Defaults
========

This module defines the default values for all parameters within DFT-D3.
"""

__all__ = [
    "D3_CN_CUTOFF",
    "D3_DISP2_CUTOFF",
    "D3_DISP3_CUTOFF",
    "D3_KCN",
    "A1",
    "A2",
    "S6",
    "S8",
    "S9",
    "RS9",
    "ALP",
    "MAX_ELEMENT",
]

# DFT-D3

# These mirror `realspace_cutoff` in s-dftd3's `dftd3_cutoff`; they differ
# because the terms they truncate decay differently, see `tad_dftd3.cutoff`.

D3_CN_CUTOFF = 40.0
"""Coordination number cutoff (40.0)."""

D3_DISP2_CUTOFF = 60.0
"""Two-body interaction cutoff (60.0)."""

D3_DISP3_CUTOFF = 40.0
"""Three-body interaction cutoff (40.0)."""

D3_KCN = 16.0
"""Steepness of counting function (16.0)."""

# DFT-D3 damping parameters

A1 = 0.4
"""Scaling for the C8 / C6 ratio in the critical radius (0.4)."""

A2 = 5.0
"""Offset parameter for the critical radius (5.0)."""

S6 = 1.0
"""Scaling factor for the C6 interaction (1.0)."""

S8 = 1.0
"""Scaling factor for the C8 interaction (1.0)."""

S9 = 1.0
"""Scaling for dispersion coefficients (1.0)."""

RS9 = 4.0 / 3.0
"""Scaling for van-der-Waals radii in damping function (4.0/3.0)."""

ALP = 14.0
"""Exponent of zero damping function (14.0)."""

# other

MAX_ELEMENT = 104
"""Atomic number (+1 for dummy) of last element supported by DFT-D3."""
