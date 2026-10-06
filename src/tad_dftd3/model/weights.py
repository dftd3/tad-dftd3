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
>>> import tad_mctc as mctc
>>> numbers = mctc.convert.symbol_to_number(["O", "H", "H"])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
... ], dtype=torch.double)
>>> ref = d3.reference.Reference.load(dtype=torch.double)
>>> structure = mctc.Structure(numbers=numbers, positions=positions)
>>> cn_model = d3.ncoord.cn_d3.replace(cutoff=d3.defaults.D3_CN_CUTOFF)
>>> cn = cn_model(structure)
>>> weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_log_weight)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> for row in c6.tolist():
...     print(" ".join(f"{v:10.7f}" for v in row))
10.4130470  5.4368823  5.4368823
 5.4368823  3.0930153  3.0930153
 5.4368823  3.0930153  3.0930153
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from tad_mctc.typing import Tensor

from ..reference import Reference

__all__ = ["WeightingFunction", "gaussian_log_weight", "weight_references"]


WeightingFunction = Callable[[Tensor], Tensor]
"""
Function that weights the reference systems by the difference of their
coordination number to the one of the atom, as the logarithm of the
(unnormalized) weight, so that :func:`weight_references` can normalize the
weights without them underflowing.
"""


def gaussian_log_weight(dcn: Tensor, factor: float = 4.0) -> Tensor:
    """
    Logarithm of the Gaussian weight of a reference system,
    ``-factor * dcn**2``, i.e. the weight ``exp(-factor * dcn**2)`` of D3.

    Parameters
    ----------
    dcn : Tensor
        Difference of coordination numbers.
    factor : float
        Steepness of the Gaussian. Defaults to 4.0.

    Returns
    -------
    Tensor
        Logarithm of the weight of each reference system.
    """
    return -factor * dcn * dcn


def weight_references(
    numbers: Tensor,
    cn: Tensor,
    reference: Reference,
    weighting_function: WeightingFunction = gaussian_log_weight,
    **kwargs: Any,
) -> Tensor:
    """
    Normalized weights of the reference systems of each atom.

    The weights are normalized as a softmax of their logarithms, which
    subtracts the largest before exponentiating. So nothing underflows, in
    any precision, also not for a coordination number far from all
    references (e.g. an atom inside a fullerene, La3N@C80), where the
    closest reference takes all the weight, and the gradient is the exact
    derivative everywhere.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system.
    cn : Tensor
        Coordination numbers for all atoms in the system.
    reference : Reference
        Reference systems for D3 model.
    weighting_function : WeightingFunction, optional
        Logarithm of the weight of a reference system, see
        :data:`WeightingFunction`. Defaults to :func:`gaussian_log_weight`.
    **kwargs : Any
        Passed on to `weighting_function`.

    Returns
    -------
    Tensor
        Weights of all reference systems, zero for the reference slots an
        element does not have and for padding atoms.
    """
    refcn = reference.cn[numbers]
    mask = refcn >= 0

    log_weights = torch.where(
        mask,
        weighting_function(refcn - cn.unsqueeze(-1), **kwargs),
        -torch.inf,
    )

    # A padding atom has no reference at all; any finite row keeps its
    # softmax finite, and it is masked below.
    has_reference = mask.any(dim=-1, keepdim=True)
    log_weights = torch.where(has_reference, log_weights, 0.0)

    weights = torch.softmax(log_weights, dim=-1)
    return torch.where(mask, weights, torch.zeros_like(weights))
