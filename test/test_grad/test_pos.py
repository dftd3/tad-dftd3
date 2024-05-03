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
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck, dgradgradcheck, jacrev
from tad_mctc.batch import pack

from tad_dftd3 import dftd3
from tad_dftd3.typing import DD, Callable, Tensor

from ..conftest import DEVICE, FAST_MODE
from .samples import samples

sample_list = ["LiH", "AmF3", "SiH4", "MB16_43_01"]

tol = 1e-8


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return dftd3(numbers, pos, param)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def gradchecker_batch(dtype: torch.dtype, name1: str, name2: str) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(pos: Tensor) -> Tensor:
        return dftd3(numbers, pos, param)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = sample["grad"].to(**dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    # automatic gradient
    energy = torch.sum(dftd3(numbers, pos, param))
    (grad,) = torch.autograd.grad(energy, pos)

    assert pytest.approx(ref.cpu(), abs=tol) == grad.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_backward(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = sample["grad"].to(**dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref = sample["grad"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    # automatic gradient
    energy = torch.sum(dftd3(numbers, positions, param))
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_functorch(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = sample["grad"].to(**dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    def dftd3_func(p: Tensor) -> Tensor:
        return dftd3(numbers, p, param).sum()

    grad = jacrev(dftd3_func)(pos)
    assert isinstance(grad, Tensor)

    assert grad.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == grad.detach().cpu()
