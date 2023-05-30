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
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from tad_dftd3 import dftd3, util
from tad_dftd3.typing import Callable, Tensor, Tuple

from ..samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01"]

tol = 1e-8


def gradchecker(
    dtype: torch.dtype, name: str
) -> Tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    param = {
        "s6": positions.new_tensor(1.00000000),
        "s8": positions.new_tensor(0.78981345),
        "s9": positions.new_tensor(1.00000000),
        "a1": positions.new_tensor(0.49484001),
        "a2": positions.new_tensor(5.73083694),
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
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> Tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    sample1, sample2 = samples[name1], samples[name2]
    numbers = util.pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    positions = util.pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )
    param = {
        "s6": positions.new_tensor(1.00000000),
        "s8": positions.new_tensor(0.78981345),
        "s9": positions.new_tensor(1.00000000),
        "a1": positions.new_tensor(0.49484001),
        "a2": positions.new_tensor(5.73083694),
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
    assert gradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


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
    assert gradgradcheck(func, diffvars, atol=tol)

    diffvars.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_autograd(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    # GFN1-xTB parameters
    param = {
        "s6": positions.new_tensor(1.00000000),
        "s8": positions.new_tensor(2.40000000),
        "s9": positions.new_tensor(0.00000000),
        "a1": positions.new_tensor(0.63000000),
        "a2": positions.new_tensor(5.00000000),
    }

    ref = sample["grad"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    # automatic gradient
    energy = torch.sum(dftd3(numbers, positions, param))
    (grad,) = torch.autograd.grad(energy, positions)

    assert pytest.approx(ref, abs=tol) == grad

    positions.detach_()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_backward(dtype: torch.dtype, name: str) -> None:
    """Compare with reference values from tblite."""
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    # GFN1-xTB parameters
    param = {
        "s6": positions.new_tensor(1.00000000),
        "s8": positions.new_tensor(2.40000000),
        "s9": positions.new_tensor(0.00000000),
        "a1": positions.new_tensor(0.63000000),
        "a2": positions.new_tensor(5.00000000),
    }

    ref = sample["grad"].type(dtype)

    # variable to be differentiated
    positions.requires_grad_(True)

    # automatic gradient
    energy = torch.sum(dftd3(numbers, positions, param))
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    assert pytest.approx(ref, abs=tol) == grad_backward

    positions.detach_()
