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
from tad_mctc.autograd import dgradcheck, dgradgradcheck, vmap_matches_loop
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import dftd3

from ..conftest import DEVICE, FAST_MODE
from ..samples import mols as samples

sample_list = ["LiH", "AmF3", "SiH4", "MB16_43_01"]

tol = 1e-8


def gradchecker(dtype: torch.dtype, name: str) -> tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor, Tensor],  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # variables to be differentiated
    param = (
        torch.tensor(1.00000000, requires_grad=True, **dd),
        torch.tensor(0.78981345, requires_grad=True, **dd),
        torch.tensor(0.49484001, requires_grad=True, **dd),
        torch.tensor(5.73083694, requires_grad=True, **dd),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs: Tensor) -> Tensor:
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        return dftd3(numbers, positions, input_param)

    return func, param


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
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor, Tensor],  # differentiable variables
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

    # variables to be differentiated
    param = (
        torch.tensor(1.00000000, requires_grad=True, **dd),
        torch.tensor(0.78981345, requires_grad=True, **dd),
        torch.tensor(0.49484001, requires_grad=True, **dd),
        torch.tensor(5.73083694, requires_grad=True, **dd),
    )
    label = ("s6", "s8", "a1", "a2")

    def func(*inputs: Tensor) -> Tensor:
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        return dftd3(numbers, positions, input_param)

    return func, param


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
def test_gradgradcheck_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


def _s9_setup(name: str) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }
    return numbers, positions, param


@pytest.mark.grad
@pytest.mark.parametrize("name", ["SiH4", "MB16_43_01"])
def test_s9_zero_autograd(name: str) -> None:
    """
    At ``s9 = 0`` the three-body term vanishes, but not its derivative: the
    energy is linear in `s9`, so the derivative is the three-body energy.
    """
    numbers, positions, param = _s9_setup(name)
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    e0 = dftd3(numbers, positions, {**param, "s9": torch.tensor(0.0, **dd)})
    e1 = dftd3(numbers, positions, {**param, "s9": torch.tensor(1.0, **dd)})

    s9 = torch.tensor(0.0, requires_grad=True, **dd)
    energy = dftd3(numbers, positions, {**param, "s9": s9}).sum()
    (grad,) = torch.autograd.grad(energy, s9)

    assert pytest.approx(e0.sum().item(), abs=tol) == energy.item()
    assert grad != 0.0
    assert pytest.approx((e1 - e0).sum().item(), abs=tol) == grad.item()


@pytest.mark.grad
@pytest.mark.parametrize("name", ["SiH4", "MB16_43_01"])
def test_s9_zero_functorch(name: str) -> None:
    """`jacrev` at ``s9 = 0`` and `vmap` over several `s9` values."""
    numbers, positions, param = _s9_setup(name)
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    def energy(s9: Tensor) -> Tensor:
        return dftd3(numbers, positions, {**param, "s9": s9}).sum()

    s9 = torch.tensor([0.0, 0.5, 1.0], **dd)
    assert vmap_matches_loop(energy, s9, atol=tol)

    grad = torch.func.jacrev(energy)(s9[0])
    e3 = energy(s9[2]) - energy(s9[0])
    assert pytest.approx(e3.item(), abs=tol) == grad.item()
