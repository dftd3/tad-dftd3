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
Evaluates a fractional coordination number for a given geometry or batch of geometries.
"""

import torch

from .typing import Tensor, CountingFunction


def coordination_number(
    numbers: Tensor,
    positions: Tensor,
    rcov: Tensor,
    counting_function: CountingFunction,
    **kwargs,
) -> Tensor:
    """
    Calculate the coordination number of each atom in the system.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system.
    positions : Tensor
        The positions of the atoms in the system.
    rcov : Tensor
        Covalent radii for all atoms in the system.
    counting_function : Callable
        Calculates counting value in range 0 to 1 from a batch of
        distances and covalent radii, additional parameters can
        be passed through via key-value arguments.

    Returns
    -------
        Tensor: The coordination number of each atom in the system.
    """
    if numbers.shape != rcov.shape:
        raise ValueError("Shape of covalent radii is not consistent with atomic numbers")
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")

    real = numbers != 0
    mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))
    distances = torch.cdist(positions, positions, p=2)
    distances[mask] = 0
    mask.diagonal(dim1=-2, dim2=-1).fill_(True)

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = counting_function(distances, rc.type(distances.dtype), **kwargs)
    cf[mask] = 0
    return torch.sum(cf, dim=-1)


def exp_count(r: Tensor, r0: Tensor, kcn: float = 16.0) -> Tensor:
    """
    Exponential counting function for coordination number contributions.

    Parameters
    ----------
    r : Tensor
        Current distance.
    r0 : Tensor
        Cutoff radius.
    kcn : float
        Steepness of the counting function.

    Returns
    -------
    Tensor
        Count of coordination number contribution.
    """
    return 1.0 / (1.0 + torch.exp(-kcn * (r0 / r - 1.0)))
