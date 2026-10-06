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
Selecting a damping by name
===========================

The name of a damping variant, as in the data base of functionals (the
``damping`` entry of ``parameters.toml``), and the two-body damping it stands
for, and the names of the three-body dampings. The two-body names follow
s-dftd3, with ``"screened"`` and ``"koide"`` of dftd. Of the three-body
names, ``"zero_d3"`` and ``"zero_d4"`` are the ATM damping of D3 and of D4
(the same averaged zero damping, on different radii), ``"zero_product"`` is
dftd's ``zero``, and the others follow dftd.
"""

from __future__ import annotations

from .base import Damping, ThreeBodyDamping, TwoBodyDamping
from .cso import CSOTwoBody
from .koide import KoideThreeBody, KoideTwoBody
from .mzero import ModifiedZeroTwoBody
from .optimizedpower import OptimizedPowerTwoBody
from .rational import RationalThreeBody, RationalTwoBody
from .screened import ScreenedThreeBody, ScreenedTwoBody
from .z import ZTwoBody
from .zero import (
    ZeroProductThreeBody,
    ZeroThreeBodyD3,
    ZeroThreeBodyD4,
    ZeroTwoBody,
)

__all__ = ["damping_from_name", "THREE_BODY_DAMPINGS", "TWO_BODY_DAMPINGS"]

TWO_BODY_DAMPINGS: tuple[tuple[str, type[TwoBodyDamping]], ...] = (
    ("rational", RationalTwoBody),
    ("zero", ZeroTwoBody),
    ("mzero", ModifiedZeroTwoBody),
    ("optimizedpower", OptimizedPowerTwoBody),
    ("cso", CSOTwoBody),
    ("z", ZTwoBody),
    ("screened", ScreenedTwoBody),
    ("koide", KoideTwoBody),
)
"""Names of the damping variants and their two-body damping."""

THREE_BODY_DAMPINGS: tuple[tuple[str, type[ThreeBodyDamping]], ...] = (
    ("zero_d3", ZeroThreeBodyD3),
    ("rational", RationalThreeBody),
    ("screened", ScreenedThreeBody),
    ("zero_d4", ZeroThreeBodyD4),
    ("zero_product", ZeroProductThreeBody),
    ("koide", KoideThreeBody),
)
"""Names of the three-body dampings."""


def damping_from_name(name: str, three_body: bool | str = True) -> Damping:
    """
    The damping of a variant.

    Parameters
    ----------
    name : str
        The two-body damping, one of the names of
        :data:`TWO_BODY_DAMPINGS`.
    three_body : bool | str, optional
        The three-body damping: one of the names of
        :data:`THREE_BODY_DAMPINGS`, ``True`` for the zero damping of D3
        (``"zero_d3"``) or ``False`` for none. Defaults to ``True``.

    Raises
    ------
    ValueError
        If a name is unknown.
    """
    if three_body is True:
        three_body = "zero_d3"

    three = None
    if three_body is not False:
        three = _lookup(THREE_BODY_DAMPINGS, three_body, "three-body damping")()

    return Damping(_lookup(TWO_BODY_DAMPINGS, name, "damping")(), three)


def _lookup(table: tuple[tuple[str, type], ...], name: str, what: str) -> type:
    """The class of `name` in `table`."""
    for key, cls in table:
        if key == name:
            return cls

    names = [key for key, _ in table]
    raise ValueError(f"Unknown {what} '{name}'; expected one of {names}.")
