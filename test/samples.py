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

Taken from :func:`tad_mctc.data.structures.get_structure`, under the names
the reference values in the ``samples.py`` modules of the test subpackages
are keyed by. Every geometry is identical to the one the references were
computed for, except ``H2O``, which is only compared against s-dftd3
computed on the fly (``test/test_disp/test_cutoff.py``).
"""

from __future__ import annotations

from typing import Any

from tad_mctc.data.structures import get_structure
from tad_mctc.typing import Molecule

__all__ = ["merge_nested_dicts", "mols"]

_SOURCES: dict[str, tuple[str, str]] = {
    "AmF3": ("other", "AmF3"),
    "C6H5I-CH3SH": ("other", "C6H5I-CH3SH"),
    "H2O": ("heavy28", "h2o"),
    "La3N@C80": ("other", "La3N@C80"),
    "LiH": ("mb16_43", "LiH"),
    "MB16_43_01": ("mb16_43", "01"),
    "PbH4-BiH3": ("heavy28", "pbh4_bih3"),
    "SiH4": ("mb16_43", "SiH4"),
}
"""Name used in this test suite -> ``(collection, record)``."""


def _molecule(collection: str, record: str) -> Molecule:
    structure = get_structure(collection, record)
    return {"numbers": structure.numbers, "positions": structure.positions}


mols: dict[str, Molecule] = {
    name: _molecule(*source) for name, source in _SOURCES.items()
}


def merge_nested_dicts(
    a: dict[str, Molecule], b: dict[str, Any]
) -> dict[str, Any]:
    """
    Add the geometry from `a` to the reference values in `b` (changed in
    place) for every name in both.

    Parameters
    ----------
    a : dict[str, Molecule]
        Geometries (not changed).
    b : dict[str, Any]
        Reference values (changed).

    Returns
    -------
    dict[str, Any]
        Merged dictionary `b`.
    """
    for key in b:
        if key in a:
            b[key].update(a[key])
    return b
