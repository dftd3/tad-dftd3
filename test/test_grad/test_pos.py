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
from tad_mctc import Structure
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor
from torch.func import jacrev

from tad_dftd3 import dftd3

from ..conftest import DEVICE, FAST_MODE
from ..reference import reference_gradient
from ..utils import load_sample, load_structure

sample_list: list[tuple[str, str]] = [
    ("mb16_43", "LiH"),
    ("other", "AmF3"),
    ("mb16_43", "SiH4"),
    ("mb16_43", "01"),
]

tol = 1e-8


def gradchecker(dtype: torch.dtype, source: tuple[str, str]) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(*source, dd)

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
        return dftd3(Structure(numbers=numbers, positions=pos), param)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", sample_list)
def test_gradcheck(dtype: torch.dtype, source: tuple[str, str]) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, source)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", sample_list)
def test_gradgradcheck(dtype: torch.dtype, source: tuple[str, str]) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, source)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def gradchecker_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack([load_sample(*src, dd)[0] for src in (source1, source2)])
    positions = pack([load_sample(*src, dd)[1] for src in (source1, source2)])
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
        return dftd3(Structure(numbers=numbers, positions=pos), param)

    return func, positions


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source1", [("mb16_43", "LiH")])
@pytest.mark.parametrize("source2", sample_list)
def test_gradcheck_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, source1, source2)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source1", [("mb16_43", "LiH")])
@pytest.mark.parametrize("source2", sample_list)
def test_gradgradcheck_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, source1, source2)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", sample_list)
def test_autograd(dtype: torch.dtype, source: tuple[str, str]) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(*source, dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref, _ = reference_gradient(load_structure(*source, dd), param)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    # automatic gradient
    energy = torch.sum(dftd3(Structure(numbers=numbers, positions=pos), param))
    (grad,) = torch.autograd.grad(energy, pos)

    assert pytest.approx(ref.cpu(), abs=tol) == grad.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", sample_list)
def test_backward(dtype: torch.dtype, source: tuple[str, str]) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(*source, dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref, _ = reference_gradient(load_structure(*source, dd), param)

    # variable to be differentiated
    positions.requires_grad_(True)

    # automatic gradient
    energy = torch.sum(
        dftd3(Structure(numbers=numbers, positions=positions), param)
    )
    energy.backward()

    assert positions.grad is not None
    grad_backward = positions.grad.clone()

    # also zero out gradients when using `.backward()`
    positions.detach_()
    positions.grad.data.zero_()

    assert pytest.approx(ref.cpu(), abs=tol) == grad_backward.cpu()


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", sample_list)
def test_functorch(dtype: torch.dtype, source: tuple[str, str]) -> None:
    """Compare with reference values from tblite."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(*source, dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref, _ = reference_gradient(load_structure(*source, dd), param)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    def dftd3_func(p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=numbers, positions=p), param).sum()

    grad = jacrev(dftd3_func)(pos)
    assert isinstance(grad, Tensor)

    assert grad.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == grad.detach().cpu()
