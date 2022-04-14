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
Implementation of D3 model to obtain atomic C6 coefficients for a given geometry.
"""

import torch

from .reference import Reference
from .typing import Tensor, WeightingFunction


def atomic_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
):
    """
    Calculate atomic dispersion coefficients.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system.
    weights : Tensor
        Weights of all reference systems.
    reference : Reference
        Reference systems for D3 model.

    Returns
    -------
    Tensor
        Atomic dispersion coefficients.
    """

    c6 = reference.c6[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    gw = torch.mul(weights.unsqueeze(-1).unsqueeze(-3), weights.unsqueeze(-2).unsqueeze(-4))

    return torch.sum(torch.sum(torch.mul(gw, c6), dim=-1), dim=-1)


def weight_references(
    numbers: Tensor,
    cn: Tensor,
    reference: Reference,
    weighting_function,
    epsilon = 1.0e-20,
    **kwargs
) -> Tensor:
    """
    Calculate the weights of the reference system.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system.
    cn : Tensor
        Coordination numbers for all atoms in the system.
    reference : Reference
        Reference systems for D3 model.
    weighting_function : Callable
        Function to calculate weight of individual reference systems.

    Returns
    -------
    Tensor
        Weights of all reference systems
    """

    mask = reference.cn[numbers] < 0

    weights = weighting_function(reference.cn[numbers] - cn.unsqueeze(-1), **kwargs)
    weights[mask] = 0
    norms = torch.add(torch.sum(weights, dim=-1), epsilon)

    return weights / norms.unsqueeze(-1)


def weight_cn(dcn: Tensor, factor: float = 4.0):
    """
    Calculate weight of indivdual reference system.

    Parameters
    ----------
    dcn : Tensor
        Difference of coordination numbers.
    factor : float
        Factor to calculate weight.

    Returns
    -------
    Tensor
        Weight of individual reference system.
    """

    return torch.exp(-factor * dcn.pow(2))
