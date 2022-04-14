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
Torch autodiff DFT-D3 implementation.
"""

import torch

from . import data, disp, ncoord, model, reference, util
from .typing import (
    Dict,
    Tensor,
    Optional,
    CountingFunction,
    WeightingFunction,
    DampingFunction,
)


def dftd3(
    numbers: Tensor,
    positions: Tensor,
    param: Dict[str, float],
    *,
    ref: Optional[reference.Reference] = None,
    rcov: Optional[Tensor] = None,
    rvdw: Optional[Tensor] = None,
    r4r2: Optional[Tensor] = None,
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = model.weight_cn,
    damping_function: DampingFunction = disp.rational_damping,
) -> Tensor:
    """
    Evaluate DFT-D3 dispersion energy for a batch of geometries.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers of the atoms in the system.
    positions : torch.Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict
        DFT-D3 damping parameters

    Other Parameters
    ----------------
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
    weighting_function : Callable
        Function to calculate weight of individual reference systems.
    counting_function : Callable
        Calculates counting value in range 0 to 1 for each atom pair.

    Returns
    -------
    torch.Tensor
        DFT-D3 dispersion energy for each geometry.
    """

    if ref is None:
        ref = reference.Reference().type(positions.dtype)
    if rcov is None:
        rcov = data.covalent_rad_d3[numbers].type(positions.dtype)
    if rvdw is None:
        rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(
            positions.dtype
        )
    if r4r2 is None:
        r4r2 = data.sqrt_z_r4_over_r2[numbers].type(positions.dtype)

    cn = ncoord.coordination_number(numbers, positions, rcov, counting_function)
    weights = model.weight_references(numbers, cn, ref, weighting_function)
    c6 = model.atomic_c6(numbers, weights, ref)
    energy = disp.dispersion(
        numbers, positions, c6, rvdw, r4r2, damping_function, **param
    )

    return energy
