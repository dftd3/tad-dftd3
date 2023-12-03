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
Coordination number: D3
=======================

Calculation of D3 coordination number.
"""
import torch

from .. import data, defaults
from .._typing import DD, Any, CountingFunction, Optional, Tensor
from ..utils import cdist, real_pairs
from .count import exp_count

__all__ = ["coordination_number_d3"]


def coordination_number_d3(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = exp_count,
    rcov: Optional[Tensor] = None,
    cutoff: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    """
    Calculate the coordination number of each atom in the system.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system.
    positions : Tensor
        The positions of the atoms in the system.
    counting_function : Callable
        Calculates counting value in range 0 to 1 from a batch of
        distances and covalent radii, additional parameters can
        be passed through via key-value arguments.
    rcov : Tensor
        Covalent radii for all atoms in the system.
    cutoff : float
        Real-space cutoff for the evaluation of counting function

    Returns
    -------
        Tensor: The coordination number of each atom in the system.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_CN_CUTOFF, **dd)
    if rcov is None:
        rcov = data.covalent_rad_d3.to(**dd)[numbers]
    if numbers.shape != rcov.shape:
        raise ValueError(
            "Shape of covalent radii is not consistent with atomic numbers"
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")

    mask = real_pairs(numbers, diagonal=False)
    distances = torch.where(
        mask,
        cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        counting_function(distances, rc.to(**dd), **kwargs),
        torch.tensor(0.0, **dd),
    )
    return torch.sum(cf, dim=-1)
