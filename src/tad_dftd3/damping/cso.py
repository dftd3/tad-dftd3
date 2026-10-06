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
CSO damping
===========

The C6-scaled only damping (Schröder et al., J. Chem. Theory Comput. 2015,
11, 3163), a reformulation of the rational damping that has no C8 term and
interpolates the C6 scaling with a sigmoid of the distance, for the two-body
term,

.. math::

    E = -\dfrac{s_6 + a_1 / \left(1 + e^{R_{\text{AB}} - a_2 R_0^{\text{AB}}}
    \right)}{R^6_{\text{AB}} + \left( a_3 R_0^{\text{AB}} + a_4 \right)^6}
    C_6^{\text{AB}}

with :math:`R_0^{\text{AB}} = \sqrt{C_8 / C_6}`. The parameters :math:`a_3`
and :math:`a_4` are ``rs6`` and ``rs8`` of the parameter set.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping, pow6
from .param import DampingParam

__all__ = ["CSOTwoBody"]


class CSOTwoBody(TwoBodyDamping):
    """
    CSO damping of the two-body term. Needs `a1`; `s6` defaults to 1.0, `a2`
    to 2.5, `rs6` (``a3``) to 0.0 and `rs8` (``a4``) to 6.25. `s8` is not
    used.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("a2", defaults.CSO_A2),
        ("rs6", defaults.CSO_RS6),
        ("rs8", defaults.CSO_RS8),
    )
    label: ClassVar[str] = "CSO damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0 = pairs.qq.sqrt()

        # 1 / (1 + exp(r - a2 r0)), without the overflow of the exponential
        sigmoid = torch.sigmoid(self.value(param, "a2") * r0 - r)
        scale = self.value(param, "s6") + self.value(param, "a1") * sigmoid
        d6 = self.value(param, "rs6") * r0 + self.value(param, "rs8")
        return scale / (pow6(r) + pow6(d6))
