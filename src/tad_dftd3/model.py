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
Dispersion model
================

Implementation of D3 model to obtain atomic C6 coefficients for a given geometry.

Examples
--------
>>> import torch
>>> import tad_dftd3 as d3
>>> numbers = d3.util.to_number(["O", "H", "H"])
>>> positions = torch.Tensor([
...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
... ])
>>> ref = d3.reference.Reference()
>>> rcov = d3.data.covalent_rad_d3[numbers]
>>> cn = d3.ncoord.coordination_number(numbers, positions, rcov, d3.ncoord.exp_count)
>>> weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_weight)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> torch.set_printoptions(precision=7)
>>> print(c6)
tensor([[10.4130471,  5.4368822,  5.4368822],
        [ 5.4368822,  3.0930154,  3.0930154],
        [ 5.4368822,  3.0930154,  3.0930154]], dtype=torch.float64)
"""
import torch

from ._typing import Any, Tensor, WeightingFunction
from .reference import Reference
from .utils import real_atoms


def atomic_c6(numbers: Tensor, weights: Tensor, reference: Reference) -> Tensor:
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
    gw = torch.mul(
        weights.unsqueeze(-1).unsqueeze(-3), weights.unsqueeze(-2).unsqueeze(-4)
    )

    return torch.sum(torch.sum(torch.mul(gw, c6), dim=-1), dim=-1)


def gaussian_weight(dcn: Tensor, factor: float = 4.0) -> Tensor:
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


def weight_references(
    numbers: Tensor,
    cn: Tensor,
    reference: Reference,
    weighting_function: WeightingFunction = gaussian_weight,
    **kwargs: Any,
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

    mask = reference.cn[numbers] >= 0

    # Due to the exponentiation, `norms` and `weights` may become very small.
    # This may cause problems for the division by `norms`. It may occur that
    # `weights` and `norms` are equal, in which case the result should be
    # exactly one. This might, however, not be the case and ultimately cause
    # larger deviations in the final values.
    #
    # This must be done in the D4 variant because the weighting functions
    # contains higher powers, which lead to values down to 1e-300.
    # Since there are also cases in D3, we have to evaluate this portion
    # in double precision to retain the correct results and avoid nan's.
    dcn = (reference.cn[numbers] - cn.unsqueeze(-1)).type(torch.double)
    weights = torch.where(
        mask,
        weighting_function(dcn, **kwargs),
        torch.tensor(0.0, device=dcn.device, dtype=dcn.dtype),  # not eps!
    )

    # Nevertheless, we must avoid zero division here in batched calculations.
    #
    # Previously, a small value was added to `norms` to prevent division by zero
    # (`norms = torch.add(torch.sum(weights, dim=-1), 1e-20)`). However, even
    # such small values can lead to relatively large deviations because the
    # small value is not added to the weights, and hence, the case where
    # `weights` and `norms` are equal does not yield one anymore. In fact, the
    # test suite fails because some elements deviate up to around 1e-4.
    #
    # We solve this issue by using a mask from the atoms and only add a small
    # value, where the actual padding zeros are.
    norms = torch.where(
        real_atoms(numbers),
        torch.sum(weights, dim=-1),
        torch.tensor(torch.finfo(dcn.dtype).eps, device=cn.device, dtype=dcn.dtype),
    )
    return (weights / norms.unsqueeze(-1)).type(cn.dtype)
