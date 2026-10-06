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
r"""
Screened rational damping
=========================

The rational (Becke-Johnson) damping with a screened radius, ported from
dftd's ``dftd_damping_screened``. The critical radius
:math:`R_\text{poly} = a_1 R_\text{damp} + a_2` is switched on by an error
function of the distance,

.. math::

    R_\text{eff} = R_\text{poly} \, \frac{1}{2} \left(1 + \operatorname{erf}
    \left(-a_3 \left(R - a_4 R_\text{damp}\right)\right)\right),

and the two-body term is damped by
:math:`(R + R_\text{eff})^{-n}` in place of :math:`R^{-n}`, the three-body
term by :math:`\prod (R / (R + R_\text{eff}))^3` over the three pairs.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .base import (
    DFTD_RADIUS_DEFAULTS,
    PairData,
    ThreeBodyDamping,
    TripleData,
    TwoBodyDamping,
    scaled_radius,
)
from .param import DampingParam

__all__ = ["ScreenedThreeBody", "ScreenedTwoBody"]


def _effective_radius(
    damping: TwoBodyDamping | ThreeBodyDamping,
    param: DampingParam,
    r: Tensor,
    rdamp: Tensor,
) -> Tensor:
    """The critical radius, switched on by the error function."""
    arg = -damping.value(param, "a3") * (r - damping.value(param, "a4") * rdamp)
    return scaled_radius(damping, param, rdamp) * 0.5 * (1.0 + torch.erf(arg))


class ScreenedTwoBody(TwoBodyDamping):
    """
    Screened rational damping of the two-body term, as dftd. Needs `s6`,
    `s8`, `a3` and `a4`; `a1` and `a2` default to 1 and 0. Reads the damping
    radius of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Screened rational damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        inv = 1.0 / (r + _effective_radius(self, param, r, pairs.rdamp))
        inv2 = inv * inv
        inv6 = inv2 * inv2 * inv2

        d6 = self.value(param, "s6") * inv6
        d8 = self.value(param, "s8") * inv6 * inv2
        return d6 + pairs.qq * d8


class ScreenedThreeBody(ThreeBodyDamping):
    """
    Screened rational damping of the three-body term, as dftd. Needs `s9`,
    `a3` and `a4`; `a1` and `a2` default to 1 and 0. Reads the damping radii
    of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Screened rational damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        def pair(r2: Tensor, rdamp: Tensor) -> Tensor:
            r = torch.sqrt(r2)
            return (r / (r + _effective_radius(self, param, r, rdamp))) ** 3

        return (
            self.value(param, "s9")
            * pair(triples.r2ij, triples.rdampij)
            * pair(triples.r2ik, triples.rdampik)
            * pair(triples.r2jk, triples.rdampjk)
        )
