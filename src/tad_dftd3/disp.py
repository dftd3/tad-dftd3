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
Dispersion energy
=================

This module provides the dispersion energy evaluation for the pairwise interactions.

Example
-------
>>> import torch
>>> import tad_dftd3 as d3
>>> numbers = torch.tensor([  # define fragments by setting atomic numbers to zero
...     [8, 1, 1, 8, 1, 6, 1, 1, 1],
...     [0, 0, 0, 8, 1, 6, 1, 1, 1],
...     [8, 1, 1, 0, 0, 0, 0, 0, 0],
... ])
>>> positions = torch.tensor([  # define coordinates once
...     [-4.224363834, +0.270465696, +0.527578960],
...     [-5.011768887, +1.780116228, +1.143194385],
...     [-2.468758653, +0.479766200, +0.982905589],
...     [+1.146167671, +0.452771215, +1.257722311],
...     [+1.841554378, -0.628298322, +2.538065200],
...     [+2.024899840, -0.438480095, -1.127412563],
...     [+1.210773578, +0.791908575, -2.550591723],
...     [+4.077073644, -0.342495506, -1.267841745],
...     [+1.404422261, -2.365753991, -1.503620411],
... ]).repeat(numbers.shape[0], 1, 1)
>>> ref = d3.reference.Reference()
>>> param = dict(a1=0.49484001, s8=0.78981345, a2=5.73083694)  # r²SCAN-D3(BJ)
>>> cn = d3.ncoord.coordination_number(numbers, positions)
>>> weights = d3.model.weight_references(numbers, cn, ref)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> energy = d3.disp.dispersion(numbers, positions, c6, **param)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(energy[0] - energy[1] - energy[2]))  # energy in Hartree
tensor(-0.0003964)
"""

import torch

from . import data
from .damping import rational_damping
from .typing import Dict, Optional, Tensor, DampingFunction
from .util import real_pairs


def dispersion(
    numbers: Tensor,
    positions: Tensor,
    param: Dict[str, float],
    c6: Tensor,
    rvdw: Optional[Tensor] = None,
    r4r2: Optional[Tensor] = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: Optional[Tensor] = None,
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
    if cutoff is None:
        cutoff = torch.tensor(50.0, dtype=positions.dtype)
    if r4r2 is None:
        r4r2 = data.sqrt_z_r4_over_r2[numbers].type(positions.dtype)
    if rvdw is None:
        rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(
            positions.dtype
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")
    if numbers.shape != r4r2.shape:
        raise ValueError(
            "Shape of expectation values is not consistent with atomic numbers"
        )

    energy = dispersion2(
        numbers, positions, param, c6, rvdw, r4r2, damping_function, cutoff
    )

    if "s9" in param and param["s9"] != 0.0:
        energy += dispersion3(
            numbers, positions, param, c6, rvdw, r4r2, damping_function, cutoff
        )

    return energy


def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: Dict[str, float],
    c6: Tensor,
    rvdw: Tensor,
    r4r2: Tensor,
    damping_function: DampingFunction,
    cutoff: Tensor,
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
    eps = torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype)
    mask = real_pairs(numbers, diagonal=False)
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        eps,
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, rvdw, qq, param),
        torch.tensor(0.0, dtype=distances.dtype),
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, rvdw, qq, param),
        torch.tensor(0.0, dtype=distances.dtype),
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    return param.get("s6", 1.0) * e6 + param.get("s8", 1.0) * e8


def dispersion3(
    numbers: Tensor,
    positions: Tensor,
    param: Dict[str, float],
    c6: Tensor,
    rvdw: Tensor,
    r4r2: Tensor,
    damping_function: DampingFunction,
    cutoff: Tensor,
) -> Tensor:
    s9 = param.get("s9", 1.0)
    rs9 = 1.0  # 4.0 / 3.0

    srvdw = rs9 * rvdw
    print(c6)

    c9 = -s9 * torch.sqrt(torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2)))
    print(c6.unsqueeze(-1) * c6.unsqueeze(-2))

    return torch.zeros_like(numbers)
