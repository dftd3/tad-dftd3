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
Koide damping
=============

The spherical-wave expanded (Koide) damping with an optional
exchange-correlation screening from the overlap of Slater densities, ported
from dftd's ``dftd_damping_koide``.

The dipole-dipole and dipole-quadrupole terms are attenuated by the
functions :math:`\chi_{11}` and :math:`\chi_{12}` of the Koide expansion, of
the distances :math:`x = s_{r,n} R / R_\text{poly}` scaled by the critical
radius :math:`R_\text{poly} = a_1 R_\text{damp} + a_2`; the three-body term
by :math:`\prod \sqrt{\chi_{11}}` over the three pairs, with
:math:`s_{r,9}`. With :math:`s_\text{xc} \neq 0`, both are scaled by
:math:`1 - s_\text{xc} S`, with the normalized overlap :math:`S` of two
Slater densities of radius :math:`r_{s,\text{xc}} R_\text{poly}` (for three
bodies, the product of the three).

Whether the screening is evaluated is a static choice, as for `s9`: not if
`sxc` is not set or the Python number zero, always for a tensor.
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

__all__ = ["KoideThreeBody", "KoideTwoBody"]


# Coefficients of the polynomials of the Koide expansion, in ascending
# order, as in dftd (`fact(n)` is n!).
_P2 = (
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1699.0 / 207360.0,
    259.0 / 207360.0,
    197.0 / 1451520.0,
    19.0 / 2177280.0,
    1.0 / 4354560.0,
)
_P0 = (
    89.0 / 13824.0,
    89.0 / 13824.0,
    119.0 / 41472.0,
    5.0 / 6912.0,
    11.0 / 103680.0,
    1.0 / 124416.0,
    1.0 / 4354560.0,
)
_P3 = (
    1.0,
    1.0,
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    109.0 / 552960.0,
    13.0 / 552960.0,
    211.0 / 96768000.0,
    57.0 / 435456000.0,
    1.0 / 290304000.0,
)
_P1 = (
    47.0 / 552960.0,
    47.0 / 552960.0,
    7.0 / 184320.0,
    1.0 / 103680.0,
    209.0 / 145152000.0,
    11.0 / 96768000.0,
    1.0 / 290304000.0,
)


def _horner(x: Tensor, coefficients: tuple[float, ...]) -> Tensor:
    """``c0 + x * (c1 + x * (c2 + ...))``, nested as in dftd."""
    value = torch.full_like(x, coefficients[-1])
    for c in reversed(coefficients[:-1]):
        value = c + x * value
    return value


def _chi11(x: Tensor) -> Tensor:
    """Dipole-dipole attenuation function of the Koide expansion."""
    exp_x = torch.exp(-x)
    phi2 = 1.0 - exp_x * _horner(x, _P2)
    phi0 = exp_x * (x * x * x * _horner(x, _P0))
    return phi2 * phi2 + 0.5 * phi0 * phi0


def _chi12(x: Tensor) -> Tensor:
    """Dipole-quadrupole attenuation function of the Koide expansion."""
    exp_x = torch.exp(-x)
    phi3 = 1.0 - exp_x * _horner(x, _P3)
    phi1 = exp_x * (x * x * x * x * x * _horner(x, _P1))
    return phi3 * phi3 + (2.0 / 3.0) * phi1 * phi1


def _slater_overlap(r: Tensor, rslater: Tensor | float) -> Tensor:
    """Normalized overlap of two Slater densities of radius `rslater`."""
    y = r / rslater
    return torch.exp(-y) * (1.0 + y + y * y / 3.0)


def _screened(param: DampingParam) -> bool:
    """Whether the XC screening is evaluated, a static choice."""
    sxc = param.sxc
    return sxc is not None and (isinstance(sxc, Tensor) or sxc != 0.0)


class KoideTwoBody(TwoBodyDamping):
    """
    Koide damping of the two-body term, as dftd. Needs `s6`, `s8`, `rs6` and
    `rs8`; `a1` and `a2` default to 1 and 0, and `rsxc` to 1 for the
    screening by `sxc`. Reads the damping radius of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        *DFTD_RADIUS_DEFAULTS,
        ("rsxc", 1.0),
    )
    label: ClassVar[str] = "Koide damping"

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        r = pairs.distances
        rpoly = scaled_radius(self, param, pairs.rdamp)

        r2 = r * r
        r6 = r2 * r2 * r2
        r8 = r6 * r2

        chi_11 = _chi11(self.value(param, "rs6") * r / rpoly)
        chi_12 = _chi12(self.value(param, "rs8") * r / rpoly)

        d6 = self.value(param, "s6") * chi_11 / r6
        d8 = self.value(param, "s8") * chi_12 / r8
        kernel = d6 + pairs.qq * d8

        if _screened(param):
            overlap = _slater_overlap(r, self.value(param, "rsxc") * rpoly)
            kernel = kernel * (1.0 - self.value(param, "sxc") * overlap)

        return kernel


class KoideThreeBody(ThreeBodyDamping):
    """
    Koide damping of the three-body term, as dftd. Needs `s9` and `rs9`; `a1`
    and `a2` default to 1 and 0, and `rsxc` to 1 for the screening by `sxc`.
    Reads the damping radii of the model.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = (
        *DFTD_RADIUS_DEFAULTS,
        ("rsxc", 1.0),
    )
    label: ClassVar[str] = "Koide damping of the three-body term"

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        rs9 = self.value(param, "rs9")
        screened = _screened(param)

        damp = torch.ones_like(triples.r)
        overlap = torch.ones_like(triples.r)
        positive = torch.ones_like(triples.r, dtype=torch.bool)
        for r2, rdamp in (
            (triples.r2ij, triples.rdampij),
            (triples.r2ik, triples.rdampik),
            (triples.r2jk, triples.rdampjk),
        ):
            r = torch.sqrt(r2)
            rpoly = scaled_radius(self, param, rdamp)
            chi = _chi11(rs9 * r / rpoly)

            # dftd returns zero if an attenuation is not positive; replaced
            # before the square root, whose derivative at zero is infinite
            positive = positive & (chi > 0.0)
            damp = damp * torch.sqrt(torch.where(chi > 0.0, chi, 1.0))

            if screened:
                rsxc = self.value(param, "rsxc")
                overlap = overlap * _slater_overlap(r, rsxc * rpoly)

        d9 = self.value(param, "s9") * damp
        if screened:
            # the product of the pair overlaps as a proxy of the triple's
            d9 = d9 * (1.0 - self.value(param, "sxc") * overlap)

        return torch.where(positive, d9, torch.zeros_like(d9))
