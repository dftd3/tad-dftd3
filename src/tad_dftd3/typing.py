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

# pylint: disable=unused-import
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import torch
from torch import Tensor

TensorOrTensors = Union[List[Tensor], Tuple[Tensor, ...], Tensor]
MaybeTensor = Union[Tensor, Optional[Tensor]]

CountingFunction = Callable[[Tensor, Tensor], Tensor]
WeightingFunction = Callable[[Tensor], Tensor]
DampingFunction = Callable[[int, Tensor, Tensor, Dict[str, Tensor]], Tensor]
Size = Union[Tuple[int], List[int], torch.Size]


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""
