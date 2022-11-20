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
Type annotations for this project.
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

Size = Optional[Union[Tuple[int], torch.Size]]
Tensor = torch.Tensor
MaybeTensor = Union[Tensor, Optional[Tensor]]
Sliceable = Union[List[Tensor], Tuple[Tensor]]

CountingFunction = Callable[[Tensor, Tensor, Any], Tensor]
WeightingFunction = Callable[[Tensor, Any], Tensor]
DampingFunction = Callable[[int, Tensor, Tensor, Any], Tensor]


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""
