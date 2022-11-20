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
from __future__ import annotations

import torch

from . import data, defaults
from .damping import dispersion_atm, rational_damping
from .typing import DampingFunction, Optional, Tensor
from .util import real_pairs


def dispersion(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: Tensor | None = None,
    **kwargs,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw : Tensor
        Van der Waals radii of the atoms in the system.
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    """
    if cutoff is None:
        cutoff = positions.new_tensor(50.0)
    if r4r2 is None:
        r4r2 = (
            data.sqrt_z_r4_over_r2[numbers].type(positions.dtype).to(positions.device)
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            "Shape of positions is not consistent with atomic numbers.",
        )
    if numbers.shape != r4r2.shape:
        raise ValueError(
            "Shape of expectation values is not consistent with atomic numbers.",
        )

    # two-body dispersion
    energy = dispersion2(
        numbers, positions, param, c6, r4r2, damping_function, cutoff, **kwargs
    )

    # three-body dispersion
    if "s9" in param and param["s9"] != 0.0:
        if rvdw is None:
            rvdw = (
                data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
                .type(positions.dtype)
                .to(positions.device)
            )

        energy += dispersion3(numbers, positions, param, c6, rvdw, cutoff)

    return energy


def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    r4r2: Tensor,
    damping_function: DampingFunction,
    cutoff: Tensor,
    **kwargs,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    """
    mask = real_pairs(numbers, diagonal=False)
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        positions.new_tensor(torch.finfo(positions.dtype).eps),
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, qq, param, **kwargs),
        positions.new_tensor(0.0),
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, qq, param, **kwargs),
        positions.new_tensor(0.0),
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", positions.new_tensor(defaults.S6))
    s8 = param.get("s8", positions.new_tensor(defaults.S8))
    return s6 * e6 + s8 * e8


def dispersion3(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    rvdw: Tensor,
    cutoff: Tensor,
    rs9: Tensor = torch.tensor(4.0 / 3.0),
) -> Tensor:
    """
    Three-body dispersion term. Currently this is only a wrapper for the
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        Dictionary of dispersion parameters. Default values are used for
        missing keys.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw : Tensor
        Van der Waals radii of the atoms in the system.
    cutoff : Tensor
        Real-space cutoff.
    rs9 : Tensor, optional
        Scaling for van-der-Waals radii in damping function. Defaults to `4.0/3.0`.

    Returns
    -------
    Tensor
        Atom-resolved three-body dispersion energy.
    """
    alp = param.get("alp", positions.new_tensor(14.0))
    s9 = param.get("s9", positions.new_tensor(14.0))
    rs9 = rs9.type(positions.dtype).to(positions.device)

    return dispersion_atm(numbers, positions, c6, rvdw, cutoff, s9, rs9, alp)
