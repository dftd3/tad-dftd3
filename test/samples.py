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
Sample geometries for this test suite.

Pulled from `tad_mctc.data.molecules.mols`, keyed by the same names, and
kept to only what `test/test_disp/test_cutoff.py` looks up here -- see
`test/reference.py`'s module docstring for why this suite keeps such
modules exactly as small as their callers need.
"""

from __future__ import annotations

from tad_mctc.data.molecules import mols
from tad_mctc.typing import Tensor

__all__ = ["samples"]

_NAMES = ["H2O", "SiH4"]

samples: dict[str, dict[str, Tensor]] = {
    name: {
        "numbers": mols[name]["numbers"],
        "positions": mols[name]["positions"],
    }
    for name in _NAMES
}
