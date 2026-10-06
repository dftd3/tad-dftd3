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
Modified zero damping
=====================

The zero damping with an offset :math:`\beta` on the distance
(Smith et al., J. Phys. Chem. Lett. 2016, 7, 2197) for the two-body term,

.. math::

    f^n_{\text{damp}}\left(R_{\text{AB}}\right) =
    \dfrac{1}{1 + 6 \left(
    \dfrac{R_{\text{AB}}}{s_{r,n} R_0^{\text{AB}}} +
    \beta R_0^{\text{AB}}
    \right)^{-\alpha_n}}

with :math:`\alpha_6 = \alpha` and :math:`\alpha_8 = \alpha + 2`. It is the
zero damping for :math:`\beta = 0`.
"""

from __future__ import annotations

from typing import ClassVar

from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping
from .param import DampingParam
from .zero import zero_like

__all__ = ["ModifiedZeroTwoBody"]


class ModifiedZeroTwoBody(TwoBodyDamping):
    """
    Modified zero damping of the two-body term. Needs `s8`, `rs6` and `bet`;
    `s6` defaults to 1.0, `rs8` to 1.0 and `alp` to
    :data:`tad_dftd3.defaults.ALP`.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("rs8", defaults.RS8),
        ("alp", defaults.ALP),
    )
    label: ClassVar[str] = "Modified zero damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r, r0 = pairs.distances, pairs.rvdw
        alp = self.value(param, "alp")
        offset = self.value(param, "bet") * r0
        t6 = (r / (self.value(param, "rs6") * r0) + offset) ** (-alp)
        t8 = (r / (self.value(param, "rs8") * r0) + offset) ** (-(alp + 2.0))
        return zero_like(self, pairs, param, t6, t8)
