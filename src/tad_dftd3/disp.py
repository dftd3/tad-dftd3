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
>>> param = dict( # r²SCAN-D3(BJ)
...     a1=torch.tensor(0.49484001),
...     s8=torch.tensor(0.78981345),
...     a2=torch.tensor(5.73083694),
... )
>>> cn = d3.ncoord.coordination_number(numbers, positions)
>>> weights = d3.model.weight_references(numbers, cn, ref)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> energy = d3.disp.dispersion(numbers, positions, param, c6)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(energy[0] - energy[1] - energy[2]))  # energy in Hartree
tensor(-0.0003964, dtype=torch.float64)
"""
from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.data import pse

from . import data, defaults, model, ncoord
from .damping import dispersion_atm, rational_damping
from .reference import Reference
from .typing import (
    DD,
    Any,
    CountingFunction,
    DampingFunction,
    Tensor,
    WeightingFunction,
)

__all__ = ["dftd3", "dispersion", "dispersion2", "dispersion3"]


def dftd3(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    *,
    ref: Reference | None = None,
    rcov: Tensor | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    cutoff: Tensor | None = None,
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = model.gaussian_weight,
    damping_function: DampingFunction = rational_damping,
    chunk_size: int | None = None,
) -> Tensor:
    """
    Evaluate DFT-D3 dispersion energy for a batch of geometries.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers of the atoms in the system.
    positions : torch.Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        DFT-D3 damping parameters.
    ref : reference.Reference, optional
        Reference C6 coefficients.
    rcov : torch.Tensor, optional
        Covalent radii of the atoms in the system.
    rvdw : torch.Tensor, optional
        Van der Waals radii of the atoms in the system.
    r4r2 : torch.Tensor, optional
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable, optional
        Damping function evaluate distance dependent contributions.
    weighting_function : Callable, optional
        Function to calculate weight of individual reference systems.
    counting_function : Callable, optional
        Calculates counting value in range 0 to 1 for each atom pair.
    chunk_size : int, optional
        Chunk size for chunked computation of huge tensors that otherwise
        create memory bottlenecks.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D3 dispersion energy for each geometry.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if torch.max(numbers) >= defaults.MAX_ELEMENT:
        raise ValueError(
            f"No D3 parameters available for Z > {defaults.MAX_ELEMENT-1} "
            f"({pse.Z2S[defaults.MAX_ELEMENT]})."
        )

    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_DISP_CUTOFF, **dd)
    if ref is None:
        ref = Reference(**dd)
    if rcov is None:
        rcov = data.COV_D3.to(**dd)[numbers]
    if rvdw is None:
        rvdw = data.VDW_D3.to(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    if r4r2 is None:
        r4r2 = data.R4R2.to(**dd)[numbers]

    cn = ncoord.cn_d3(
        numbers, positions, counting_function=counting_function, rcov=rcov
    )
    weights = model.weight_references(numbers, cn, ref, weighting_function)
    c6 = model.atomic_c6(numbers, weights, ref, chunk_size=chunk_size)

    return dispersion(
        numbers,
        positions,
        param,
        c6,
        rvdw,
        r4r2,
        damping_function,
        cutoff=cutoff,
    )


def dispersion(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: Tensor | None = None,
    **kwargs: Any,
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

    Returns
    -------
    Tensor
        Atom-resolved DFT-D3 dispersion energy for each geometry.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_DISP_CUTOFF, **dd)
    if r4r2 is None:
        r4r2 = data.R4R2.to(**dd)[numbers]

    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            "Shape of positions is not consistent with atomic numbers.",
        )
    if numbers.shape != r4r2.shape:
        raise ValueError(
            "Shape of expectation values is not consistent with atomic numbers.",
        )
    if torch.max(numbers) >= defaults.MAX_ELEMENT:
        raise ValueError(
            f"No D3 parameters available for Z > {defaults.MAX_ELEMENT - 1} "
            f"({pse.Z2S[defaults.MAX_ELEMENT]})."
        )

    # two-body dispersion
    energy = dispersion2(
        numbers, positions, param, c6, r4r2, damping_function, cutoff, **kwargs
    )

    # three-body dispersion
    if "s9" in param and param["s9"] != 0.0:
        if rvdw is None:
            rvdw = data.VDW_D3.to(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]

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
    **kwargs: Any,
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
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, qq, param, **kwargs),
        torch.tensor(0.0, **dd),
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, qq, param, **kwargs),
        torch.tensor(0.0, **dd),
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))
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
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    alp = param.get("alp", torch.tensor(14.0, **dd))
    s9 = param.get("s9", torch.tensor(1.0, **dd))
    rs9 = rs9.type(positions.dtype).to(positions.device)

    return dispersion_atm(numbers, positions, c6, rvdw, cutoff, s9, rs9, alp)
