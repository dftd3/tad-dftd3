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
Axilrod-Teller-Muto (ATM) dispersion term
=========================================

This module provides the dispersion energy evaluation for the three-body
Axilrod-Teller-Muto dispersion term.

.. math::

    E_\text{disp}^{(3), \text{ATM}} &=
    \sum_\text{ABC} E^{\text{ABC}} f_\text{damp}\left(\overline{R}_\text{ABC}\right) \\
    E^{\text{ABC}} &=
    \dfrac{C^{\text{ABC}}_9
    \left(3 \cos\theta_\text{A} \cos\theta_\text{B} \cos\theta_\text{C} + 1 \right)}
    {\left(r_\text{AB} r_\text{BC} r_\text{AC} \right)^3} \\
    f_\text{damp} &=
    \dfrac{1}{1+ 6 \left(\overline{R}_\text{ABC}\right)^{-16}}
"""
import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples

from .. import defaults
from ..typing import DD, Tensor

__all__ = ["dispersion_atm"]


def dispersion_atm(
    numbers: Tensor,
    positions: Tensor,
    c6: Tensor,
    rvdw: Tensor,
    cutoff: Tensor,
    s9: Tensor = torch.tensor(defaults.S9),
    rs9: Tensor = torch.tensor(defaults.RS9),
    alp: Tensor = torch.tensor(defaults.ALP),
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw : Tensor
        Van der Waals radii of the atoms in the system.
    cutoff : Tensor
        Real-space cutoff.
    s9 : Tensor, optional
        Scaling for dispersion coefficients. Defaults to `1.0`.
    rs9 : Tensor, optional
        Scaling for van-der-Waals radii in damping function. Defaults to `4.0/3.0`.
    alp : Tensor, optional
        Exponent of zero damping function. Defaults to `14.0`.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    s9 = s9.type(positions.dtype).to(positions.device)
    rs9 = rs9.type(positions.dtype).to(positions.device)
    alp = alp.type(positions.dtype).to(positions.device)

    cutoff2 = cutoff * cutoff
    srvdw = rs9 * rvdw

    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_self=True)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    one = torch.tensor(1.0, **dd)

    # C9_ABC = s9 * sqrt(|C6_AB * C6_AC * C6_BC|)
    c9 = s9 * storch.sqrt(
        torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3))
    )

    r0ij = srvdw.unsqueeze(-1)
    r0ik = srvdw.unsqueeze(-2)
    r0jk = srvdw.unsqueeze(-3)
    r0 = r0ij * r0ik * r0jk

    # actually faster than other alternatives
    # very slow: (pos.unsqueeze(-2) - pos.unsqueeze(-3)).pow(2).sum(-1)
    distances = torch.pow(
        torch.where(
            mask_pairs,
            storch.cdist(positions, positions, p=2),
            eps,
        ),
        2.0,
    )

    r2ij = distances.unsqueeze(-1)
    r2ik = distances.unsqueeze(-2)
    r2jk = distances.unsqueeze(-3)
    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    # add epsilon to avoid zero division later
    r3 = torch.where(mask_triples, r1 * r2, eps)
    r5 = torch.where(mask_triples, r2 * r3, eps)

    # dividing by tiny numbers leads to huge numbers, which result in NaN's
    # upon exponentiation in the subsequent step
    mask = real_triples(numbers, mask_self=True)
    base = r0 / torch.where(mask_triples, r1, one)

    # to fix the previous mask, we mask again (not strictly necessary because
    # `ang` is also masked and we later multiply with `ang`)
    fdamp = torch.where(
        mask_triples,
        1.0 / (1.0 + 6.0 * base ** ((alp + 2.0) / 3.0)),
        zero,
    )

    s = torch.where(
        mask,
        (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik),
        zero,
    )

    ang = torch.where(
        mask_triples * (r2ij <= cutoff2) * (r2jk <= cutoff2) * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        torch.tensor(0.0, **dd),
    )

    energy = ang * fdamp * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0
