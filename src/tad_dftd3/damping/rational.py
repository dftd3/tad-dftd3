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
Rational (Becke-Johnson) damping function
=========================================

This module defines the rational damping function, also known as Becke-Johnson
damping, of the two-body term.

.. math::

    f^n_{\text{damp}}\left(R_0^{\text{AB}}\right) =
    \dfrac{R^n_{\text{AB}}}{R^n_{\text{AB}} +
    \left( a_1 R_0^{\text{AB}} + a_2 \right)^n}

The three-body damping of dftd, ``RationalThreeBody``, is the product of
this function for :math:`n = 3` over the three pairs of a triple, on the
damping radii of the model.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .. import defaults
from .base import (
    DFTD_RADIUS_DEFAULTS,
    PairData,
    ThreeBodyDamping,
    TripleData,
    TwoBodyDamping,
    pow6_pow8,
    scaled_radius,
)
from .param import DampingParam

__all__ = ["RationalThreeBody", "RationalTwoBody"]


class RationalTwoBody(TwoBodyDamping):
    """
    Rational (Becke-Johnson) damping of the two-body term. Needs `s8`, `a1`
    and `a2`; `s6` defaults to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (("s6", defaults.S6),)
    label: ClassVar[str] = "Rational damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        s6, s8 = self.value(param, "s6"), self.value(param, "s8")
        r0 = scaled_radius(self, param, pairs.rdamp)
        r6, r8 = pow6_pow8(r)
        r0_6, r0_8 = pow6_pow8(r0)
        return s6 / (r6 + r0_6) + s8 * pairs.qq / (r8 + r0_8)


class RationalThreeBody(ThreeBodyDamping):
    """
    Rational damping of the three-body term, as dftd. Needs `s9`; `a1` and
    `a2` default to 1 and 0. Reads the damping radii of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Rational damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        def pair(r2: Tensor, rdamp: Tensor) -> Tensor:
            r3 = r2 * torch.sqrt(r2)
            rpoly = scaled_radius(self, param, rdamp)
            return r3 / (r3 + rpoly**3)

        return (
            self.value(param, "s9")
            * pair(triples.r2ij, triples.rdampij)
            * pair(triples.r2ik, triples.rdampik)
            * pair(triples.r2jk, triples.rdampjk)
        )
