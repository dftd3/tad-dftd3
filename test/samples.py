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

`tad_mctc.data.molecules.mols` (a plain ``dict[str, dict[str, Tensor]]``)
was replaced by `tad_mctc.data.structures` (bespoke entries, `Structure`
instances) and `tad_mctc.data.structures.mstore.get_structure` (records
with a verified origin in https://github.com/grimme-lab/mstore, reached by
dataset and mstore's own bare record id, not by any locally-invented
alias). `samples` below re-flattens both sources back into the
``{"numbers": ..., "positions": ...}`` shape every test in this suite
already expects, so only this one module needs to know where each name
now lives.

Every mstore-sourced entry below was checked (`torch.equal`, not just
`torch.allclose`) against the geometry the old `mols` dict held under the
same name before it was removed, except `"CH4"`, `"H2O"` and `"NH3"`:
those three used to be bespoke, locally-authored geometries that were
*not* mstore records and turned out to be measurably different from
mstore's own `mb16_43/CH4`, `heavy28/h2o`, `heavy28/nh3` -- so this module
picks up the genuine mstore geometry under those names instead. This is
safe for every test that reaches these three names
(`test_disp/test_dftd3.py`'s live sweep against s-dftd3): every comparison
there computes its s-dftd3 reference from whatever geometry it looks up
here, live, rather than against a pre-generated fixture keyed by name, so
a different (but still valid) geometry under the same name cannot make a
comparison pass or fail incorrectly. Do not add a name here that a
fixture-based reference (`test/references/*.json`) also keys by, without
re-checking that assumption still holds.
"""

from __future__ import annotations

from tad_mctc.data.structures import structures as _other
from tad_mctc.data.structures.mstore import get_structure
from tad_mctc.typing import Tensor

__all__ = ["samples"]

# name -> (mstore collection, mstore record id), for every name this test
# suite looks up that no longer lives in `_other` (see module docstring).
_MSTORE_NAMES: dict[str, tuple[str, str]] = {
    "LiH": ("mb16_43", "LiH"),
    "H2": ("mb16_43", "H2"),
    "S2": ("mb16_43", "S2"),
    "CH4": ("mb16_43", "CH4"),
    "H2O": ("heavy28", "h2o"),
    "NH3": ("heavy28", "nh3"),
    "SiH4": ("mb16_43", "SiH4"),
    "PbH4-BiH3": ("heavy28", "pbh4_bih3"),
    "MB16_43_01": ("mb16_43", "01"),
    "MB16_43_02": ("mb16_43", "02"),
    "MB16_43_03": ("mb16_43", "03"),
    "MB16_43_07": ("mb16_43", "07"),
    "MB16_43_08": ("mb16_43", "08"),
}

samples: dict[str, dict[str, Tensor]] = {
    name: {"numbers": structure.numbers, "positions": structure.positions}
    for name, structure in _other.items()
}
samples.update(
    {
        name: {
            "numbers": (structure := get_structure(*where)).numbers,
            "positions": structure.positions,
        }
        for name, where in _MSTORE_NAMES.items()
    }
)
