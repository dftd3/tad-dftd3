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
Z damping
=========

The atomic number damping of the XDM(Z) model for the two-body term,

.. math::

    \dfrac{1}{R^n_{\text{AB}} + \dfrac{a_1}{Z_{\text{A}} + Z_{\text{B}}}
    C_n^{\text{AB}}}

with the atomic numbers :math:`Z`. There are no D3(Z) parameters in the
database of functionals.

.. note::

    s-dftd3 evaluates this with the index of the species of an atom (the
    elements in order of first appearance) in place of its atomic number, so
    its energy depends on the order of the atoms. The atomic number is used
    here, which a data-dependent index could not be under ``torch.func``
    and ``torch.compile``. Both agree if the elements of a structure are
    H, He, Li, ... in order of appearance, as in the tests.
"""

from __future__ import annotations

from typing import ClassVar

from tad_mctc.typing import Tensor

from .. import defaults
from .base import PairData, TwoBodyDamping, pow6_pow8
from .param import DampingParam

__all__ = ["ZTwoBody"]


class ZTwoBody(TwoBodyDamping):
    """
    Z damping of the two-body term. Needs `a1`; `s6` and `s8` default to 1.0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("s8", defaults.S8),
    )
    label: ClassVar[str] = "Z damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        r0c6 = self.value(param, "a1") / pairs.znum * pairs.c6
        r6, r8 = pow6_pow8(r)
        return self.value(param, "s6") / (r6 + r0c6) + self.value(
            param, "s8"
        ) * pairs.qq / (r8 + r0c6 * pairs.qq)
