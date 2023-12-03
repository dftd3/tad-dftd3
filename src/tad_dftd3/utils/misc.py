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
Miscellaneous functions
=======================

Utilities for working with tensors as well as translating between element
symbols and atomic numbers.
"""
import torch

from .._typing import List, Optional, Size, Tensor, TensorOrTensors, Union
from ..constants import PSE

__all__ = [
    "real_atoms",
    "real_pairs",
    "real_triples",
    "pack",
    "to_number",
    "get_default_device",
    "get_default_dtype",
]


def real_atoms(numbers: Tensor) -> Tensor:
    """
    Create a mask for atoms, discerning padding and actual atoms.
    Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.

    Returns
    -------
    Tensor
        Mask for atoms that discerns padding and real atoms.
    """
    return numbers != 0


def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    diagonal : bool, optional
        Flag for also writing `False` to the diagonal, i.e., to all pairs
        with the same indices. Defaults to `False`, i.e., writing False
        to the diagonal.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


def real_triples(
    numbers: torch.Tensor, diagonal: bool = False, self: bool = True
) -> Tensor:
    """
    Create a mask for triples from atomic numbers. Padding value is zero.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers for all atoms.
    diagonal : bool, optional
        Flag for also writing `False` to the space diagonal, i.e., to all
        triples with the same indices. Defaults to `False`, i.e., writing False
        to the diagonal.
    self : bool, optional
        Flag for also writing `False` to all triples where at least two indices
        are identical. Defaults to `True`, i.e., not writing `False`.

    Returns
    -------
    Tensor
        Mask for triples.
    """
    real = real_pairs(numbers, diagonal=True)
    mask = real.unsqueeze(-3) * real.unsqueeze(-2) * real.unsqueeze(-1)

    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))

    if self is False:
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-2)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-1)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-2, dim2=-1)

    return mask


def pack(
    tensors: TensorOrTensors,
    axis: int = 0,
    value: Union[int, float] = 0,
    size: Optional[Size] = None,
) -> Tensor:
    """
    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Parameters
    ----------
    tensors : list[Tensor] | tuple[Tensor] | Tensor
        List of tensors to be packed, all with identical dtypes.
    axis : int
        Axis along which tensors should be packed; 0 for first axis -1
        for the last axis, etc. This will be a new dimension.
    value : int | float
        The value with which the tensor is to be padded.
    size :
        Size of each dimension to which tensors should be padded.
        This to the largest size encountered along each dimension.

    Returns
    -------
    padded : Tensor
        Input tensors padded and packed into a single tensor.
    """
    if isinstance(tensors, Tensor):
        return tensors

    _count = len(tensors)
    _device = tensors[0].device
    _dtype = tensors[0].dtype

    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values.tolist()
    assert size is not None

    padded = torch.full((_count, *size), value, dtype=_dtype, device=_device)

    for n, source in enumerate(tensors):
        padded[(n, *[slice(0, s) for s in source.shape])] = source

    if axis != 0:
        axis = padded.dim() + 1 + axis if axis < 0 else axis
        order = list(range(1, padded.dim()))
        order.insert(axis, 0)
        padded = padded.permute(order)

    return padded


def to_number(symbols: List[str]) -> Tensor:
    """
    Obtain atomic numbers from element symbols.
    """
    return torch.flatten(
        torch.tensor([PSE.get(symbol.capitalize(), 0) for symbol in symbols])
    )


def get_default_device() -> torch.device:
    """Default device for tensors."""
    return torch.tensor(1.0).device


def get_default_dtype() -> torch.dtype:
    """Default data type for floating point tensors."""
    return torch.tensor(1.0).dtype
