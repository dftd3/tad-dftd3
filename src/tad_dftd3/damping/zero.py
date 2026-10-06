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
Zero damping
============

The zero damping of DFT-D3, which switches the dispersion off at short
distances (the damping function is zero at zero distance, unlike the
rational damping), on the van-der-Waals radii of the elements.

For the two-body term,

.. math::

    f^n_{\text{damp}}\left(R_{\text{AB}}\right) =
    \dfrac{1}{1 + 6 \left(
    \dfrac{s_{r,n} R_0^{\text{AB}}}{R_{\text{AB}}}
    \right)^{\alpha_n}}

with :math:`\alpha_6 = \alpha` and :math:`\alpha_8 = \alpha + 2`. For the
Axilrod-Teller-Muto term, the same function of the averages over the three
pairs of a triple,

.. math::

    f_\text{damp}^{(3)} = s_9 \left/ \left(1 + 6 \left(
    \dfrac{\prod s_{r,9} R_0}{\prod R}
    \right)^{(\alpha + 2) / 3}\right)\right.,

the only three-body damping of s-dftd3, where every variant uses it
(``ZeroThreeBodyD3``). It is ``zero_avg`` of dftd on the van-der-Waals radii
of D3 scaled by :math:`s_{r,9}` (in place of :math:`a_1 R_\text{damp} +
a_2`), with :math:`\alpha + 2` in place of :math:`\alpha`.

The two three-body zero dampings of dftd are ported as they are, on the
damping radii of the model, :math:`R_\text{poly} = a_1 R_\text{damp} + a_2`:
``ZeroThreeBodyD4`` (dftd's ``zero_avg``),

.. math::

    f_\text{damp}^{(3)} = s_9 \left/ \left(1 + 6 \left(
    \dfrac{\prod R_\text{poly}}{\prod R}
    \right)^{\alpha / 3}\right)\right.,

and ``ZeroProductThreeBody`` (dftd's ``zero``), the product of the two-body
zero damping of the three pairs,

.. math::

    f_\text{damp}^{(3)} = s_9 \prod \left/ \left(1 + 6 \left(
    \dfrac{s_{r,9} R_\text{poly}}{R}
    \right)^{\alpha}\right)\right..

Neither has defaults for :math:`s_{r,9}` or :math:`\alpha`, which differ
from those of D3.
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
    scaled_radius,
)
from .param import DampingParam

__all__ = [
    "averaged_zero_damping",
    "zero_like",
    "ZeroProductThreeBody",
    "ZeroThreeBodyD3",
    "ZeroThreeBodyD4",
    "ZeroTwoBody",
]


def zero_like(
    damping: TwoBodyDamping,
    pairs: PairData,
    param: DampingParam,
    t6: Tensor,
    t8: Tensor,
) -> Tensor:
    """Assemble the two-body energy kernel from the zero-damping terms."""
    r2 = pairs.distances * pairs.distances
    r6 = r2 * r2 * r2
    return damping.value(param, "s6") / (r6 * (1.0 + 6.0 * t6)) + damping.value(
        param, "s8"
    ) * pairs.qq / (r6 * r2 * (1.0 + 6.0 * t8))


class ZeroTwoBody(TwoBodyDamping):
    """
    Zero damping of the two-body term. Needs `s8` and `rs6`; `s6` defaults to
    1.0, `rs8` to 1.0 and `alp` to :data:`tad_dftd3.defaults.ALP`.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("s6", defaults.S6),
        ("rs8", defaults.RS8),
        ("alp", defaults.ALP),
    )
    label: ClassVar[str] = "Zero damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r, r0 = pairs.distances, pairs.rvdw
        alp = self.value(param, "alp")
        t6 = (self.value(param, "rs6") * r0 / r) ** alp
        t8 = (self.value(param, "rs8") * r0 / r) ** (alp + 2.0)
        return zero_like(self, pairs, param, t6, t8)


def averaged_zero_damping(
    r0: Tensor, r: Tensor, s9: Tensor | float, exponent: Tensor | float
) -> Tensor:
    """
    Zero damping of triples on the averages over their pairs,
    ``s9 / (1 + 6 (r0 / r)**exponent)``, with `r0` and `r` the products of
    the radii and of the distances of the three pairs.

    The ATM damping of both s-dftd3 and dftd4, which differ only in the
    radii and in how `alp` gives the exponent, see :class:`ZeroThreeBodyD3`
    and :class:`ZeroThreeBodyD4`.
    """
    return s9 / (1.0 + 6.0 * (r0 / r) ** exponent)


class ZeroThreeBodyD3(ThreeBodyDamping):
    """
    Zero damping of the three-body term of D3, as s-dftd3, the default: the
    averaged zero damping on the van-der-Waals radii scaled by `rs9`, with
    the exponent ``(alp + 2) / 3``. Needs `s9`; `rs9` and `alp` default to
    :data:`tad_dftd3.defaults.RS9` and :data:`tad_dftd3.defaults.ALP`.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        ("rs9", defaults.RS9),
        ("alp", defaults.ALP),
    )
    label: ClassVar[str] = "Zero damping of the three-body term (D3)"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9 = self.value(param, "rs9")
        r0 = (
            (rs9 * triples.rvdwij)
            * (rs9 * triples.rvdwik)
            * (rs9 * triples.rvdwjk)
        )
        alp = self.value(param, "alp")
        return averaged_zero_damping(
            r0, triples.r, self.value(param, "s9"), (alp + 2.0) / 3.0
        )


class ZeroThreeBodyD4(ThreeBodyDamping):
    """
    Zero damping of the three-body term of D4, as dftd4 and dftd's
    ``zero_avg``: the averaged zero damping on the damping radii of the model
    scaled to ``a1 * rdamp + a2``, with the exponent ``alp / 3``. The suffix
    names the convention, not the model: in tad-dftd3 it runs on the C6 of
    D3 (whose damping radius ``sqrt(3 r4r2_i r4r2_j)`` is the one of D4). Needs `s9` and `alp` (16 in dftd4); `a1` and `a2`
    default to 1 and 0.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Zero damping of the three-body term (D4)"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rpoly = (
            scaled_radius(self, param, triples.rdampij)
            * scaled_radius(self, param, triples.rdampik)
            * scaled_radius(self, param, triples.rdampjk)
        )
        alp = self.value(param, "alp")
        return averaged_zero_damping(
            rpoly, triples.r, self.value(param, "s9"), alp / 3.0
        )


class ZeroProductThreeBody(ThreeBodyDamping):
    """
    Zero damping of the three-body term as the product over the three pairs,
    as dftd's ``zero``. Needs `s9`, `rs9` and `alp`; `a1` and `a2` default to
    1 and 0. Reads the damping radii of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = DFTD_RADIUS_DEFAULTS
    label: ClassVar[str] = "Product zero damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9, alp = self.value(param, "rs9"), self.value(param, "alp")

        def pair(r2: Tensor, rdamp: Tensor) -> Tensor:
            rpoly = scaled_radius(self, param, rdamp)
            t = (rs9 * rpoly / torch.sqrt(r2)) ** alp
            return 1.0 / (1.0 + 6.0 * t)

        return (
            self.value(param, "s9")
            * pair(triples.r2ij, triples.rdampij)
            * pair(triples.r2ik, triples.rdampik)
            * pair(triples.r2jk, triples.rdampjk)
        )
