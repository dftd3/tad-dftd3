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
This module provides the dispersion energy evaluation for the pairwise interactions.
"""

import torch

from .typing import Tensor


def dispersion(
    numbers: Tensor,
    positions: Tensor,
    c6: Tensor,
    rvdw: Tensor,
    r4r2: Tensor,
    damping_function,
    s6: float = 1.0,
    s8: float = 1.0,
    **kwargs
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

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
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    s6 : float
        Scaling factor for the C6 interaction.
    s8 : float
        Scaling factor for the C8 interaction.
    """

    real = numbers != 0
    mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))
    distances = torch.cdist(positions, positions, p=2)
    distances[mask] = 0
    mask.diagonal(dim1=-2, dim2=-1).fill_(True)

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = damping_function(6, distances, rvdw, qq, **kwargs)
    t8 = damping_function(8, distances, rvdw, qq, **kwargs)
    t6[mask] = 0
    t8[mask] = 0

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    return s6 * e6 + s8 * e8


def rational_damping(
    order: int,
    distances: Tensor,
    rvdw: Tensor,
    qq: Tensor,
    a1: float = 0.4,
    a2: float = 5.0,
) -> Tensor:
    """
    Rational damped dispersion interaction between pairs

    Parameters
    ----------
    order : int
        Order of the dispersion interaction, e.g.
        6 for dipole-dipole, 8 for dipole-quadrupole and so on.
    distances : Tensor
        Pairwise distances between atoms in the system.
    rvdw : Tensor
        Van der Waals radii of the atoms in the system.
    qq : Tensor
        Quotient of C8 and C6 dispersion coefficients.
    a1 : float
        Scaling for the C8 / C6 ratio in the critical radius.
    a2 : float
        Offset parameter for the critical radius.
    """

    return 1.0 / (distances.pow(order) + (a1 * torch.sqrt(qq) + a2).pow(order))
