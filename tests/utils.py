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
Collection of utility functions for testing.
"""

import torch

from tad_dftd3.typing import Dict, Size, Tensor


def merge_nested_dicts(a: Dict[str, Dict], b: Dict[str, Dict]) -> Dict:  # type: ignore[type-arg]
    """
    Merge nested dictionaries. Dictionary `a` remains unaltered, while
    the corresponding keys of it are added to `b`.

    Parameters
    ----------
    a : dict
        First dictionary (not changed).
    b : dict
        Second dictionary (changed).

    Returns
    -------
    dict
        Merged dictionary `b`.
    """
    for key in b:
        if key in a:
            b[key].update(a[key])
    return b


def get_device_from_str(s: str) -> torch.device:
    """
    Convert device name to `torch.device`. Critically, this also sets the index
    for CUDA devices to `torch.cuda.current_device()`.

    Parameters
    ----------
    s : str
        Name of the device as string.

    Returns
    -------
    torch.device
        Device as torch class.

    Raises
    ------
    KeyError
        Unknown device name is given.
    """
    d = {
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda", index=torch.cuda.current_device()),
    }

    if s not in d:
        raise KeyError(f"Unknown device '{s}' given.")

    return d[s]


def reshape_fortran(x: Tensor, shape: Size) -> Tensor:
    """
    Implements Fortran's `reshape` function (column-major).

    Parameters
    ----------
    x : Tensor
        Input tensor
    shape : Size
        Output size to which `x` is reshaped.

    Returns
    -------
    Tensor
        Reshaped tensor of size `shape`.
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
