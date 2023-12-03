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
Coordination number: Counting functions
=======================================

This module contains the exponential and the error function counting functions
for the determination of the coordination number.

Only the exponential counting function is used within the D3 model.
Additionally, the analytical derivatives for the counting functions is also
provided and can be used for checking the autograd results.
"""
import torch

from .. import defaults
from .._typing import Tensor

__all__ = ["exp_count", "dexp_count"]


def exp_count(r: Tensor, r0: Tensor, kcn: float = defaults.D3_KCN) -> Tensor:
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


def dexp_count(r: Tensor, r0: Tensor, kcn: float = defaults.D3_KCN) -> Tensor:
    """
    Derivative of the exponential counting function w.r.t. the distance.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B).
    kcn : float, optional
        Steepness of the counting function. Defaults to `defaults.D4_KCN`.

    Returns
    -------
    Tensor
        Derivative of count of coordination number contribution.
    """
    expterm = torch.exp(-kcn * (r0 / r - 1.0))
    return (-kcn * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))
