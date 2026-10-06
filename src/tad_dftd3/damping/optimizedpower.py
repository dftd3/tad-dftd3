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
r"""
Optimized power damping
=======================

The rational damping with an additional zero-damping like power
:math:`\beta` (Witte et al., J. Chem. Theory Comput. 2017, 13, 2043) for the
two-body term,

.. math::

    \dfrac{R_{\text{AB}}^\beta}{R^{n + \beta}_{\text{AB}} +
    \left( a_1 R_0^{\text{AB}} + a_2 \right)^{n}
    \left( a_1 R_0^{\text{AB}} + a_2 \right)^{\beta}}

with the critical radius :math:`R_0^{\text{AB}}` of the rational damping.
It is the rational damping for :math:`\beta = 0`.
"""

from __future__ import annotations

from typing import ClassVar

from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping, pow6_pow8, scaled_radius
from .param import DampingParam

__all__ = ["OptimizedPowerTwoBody"]


class OptimizedPowerTwoBody(TwoBodyDamping):
    """
    Optimized power damping of the two-body term. Needs `s8`, `a1`, `a2` and
    `bet`; `s6` defaults to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (("s6", defaults.S6),)
    label: ClassVar[str] = "Optimized power damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        bet = self.value(param, "bet")
        r0 = scaled_radius(self, param, pairs.rdamp)
        rb = r**bet
        ab = r0**bet
        r6, r8 = pow6_pow8(r)
        r0_6, r0_8 = pow6_pow8(r0)
        return self.value(param, "s6") * rb / (
            rb * r6 + ab * r0_6
        ) + self.value(param, "s8") * pairs.qq * rb / (rb * r8 + ab * r0_8)
