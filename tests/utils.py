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
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from tad_dftd3._typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Size,
    Tensor,
    TensorOrTensors,
    Union,
)

from .conftest import FAST_MODE


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


class _GradcheckFunction(Protocol):
    """
    Type annotation for gradcheck function.
    """

    def __call__(  # type: ignore
        self,
        func: Callable[..., TensorOrTensors],
        inputs: TensorOrTensors,
        *,
        eps: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        raise_exception: bool = True,
        check_sparse_nnz: bool = False,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
        check_batched_grad: bool = False,
        check_batched_forward_grad: bool = False,
        check_forward_ad: bool = False,
        check_backward_ad: bool = True,
        fast_mode: bool = False,
    ) -> bool:
        ...


class _GradgradcheckFunction(Protocol):
    """
    Type annotation for gradgradcheck function.
    """

    def __call__(  # type: ignore
        self,
        func: Callable[..., TensorOrTensors],
        inputs: TensorOrTensors,
        grad_outputs: Optional[TensorOrTensors] = None,
        *,
        eps: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        gen_non_contig_grad_outputs: bool = False,
        raise_exception: bool = True,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
        check_batched_grad: bool = False,
        check_fwd_over_rev: bool = False,
        check_rev_over_rev: bool = True,
        fast_mode: bool = False,
    ) -> bool:
        ...


def _wrap_gradcheck(
    gradcheck_func: Union[_GradcheckFunction, _GradgradcheckFunction],
    func: Callable[..., TensorOrTensors],
    diffvars: TensorOrTensors,
    **kwargs: Any,
) -> bool:
    fast_mode = kwargs.pop("fast_mode", FAST_MODE)
    try:
        assert gradcheck_func(func, diffvars, fast_mode=fast_mode, **kwargs)
    finally:
        if isinstance(diffvars, Tensor):
            diffvars.detach_()
        else:
            for diffvar in diffvars:
                diffvar.detach_()

    return True


def dgradcheck(
    func: Callable[..., TensorOrTensors], diffvars: TensorOrTensors, **kwargs: Any
) -> bool:
    """
    Wrapper for `torch.autograd.gradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradcheck, func, diffvars, **kwargs)


def dgradgradcheck(
    func: Callable[..., TensorOrTensors], diffvars: TensorOrTensors, **kwargs: Any
) -> bool:
    """
    Wrapper for `torch.autograd.gradgradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradgradcheck, func, diffvars, **kwargs)
