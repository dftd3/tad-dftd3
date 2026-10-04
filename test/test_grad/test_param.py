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
from tad_mctc.autograd import dgradcheck, dgradgradcheck, vmap_matches_loop
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import dftd3

from ..conftest import DEVICE, FAST_MODE
from ..utils import load_sample

sample_list: list[tuple[str, str]] = [
    ("mb16_43", "LiH"),
    ("other", "AmF3"),
    ("mb16_43", "SiH4"),
    ("mb16_43", "01"),
]

tol = 1e-8


def gradchecker(dtype: torch.dtype, source: tuple[str, str]) -> tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor, Tensor],  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(*source, dd)

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
        return dftd3(
            Structure(numbers=numbers, positions=positions), input_param
        )

    return func, param


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
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor, Tensor],  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack([load_sample(*src, dd)[0] for src in (source1, source2)])
    positions = pack([load_sample(*src, dd)[1] for src in (source1, source2)])

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
        return dftd3(
            Structure(numbers=numbers, positions=positions), input_param
        )

    return func, param


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


def _s9_setup(
    source: tuple[str, str],
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers, positions = load_sample(*source, dd)
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }
    return numbers, positions, param


@pytest.mark.grad
@pytest.mark.parametrize("source", [("mb16_43", "SiH4"), ("mb16_43", "01")])
def test_s9_zero_autograd(source: tuple[str, str]) -> None:
    """
    At ``s9 = 0`` the three-body term vanishes, but not its derivative: the
    energy is linear in `s9`, so the derivative is the three-body energy.
    """
    numbers, positions, param = _s9_setup(source)
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    structure = Structure(numbers=numbers, positions=positions)
    e0 = dftd3(
        structure,
        {**param, "s9": torch.tensor(0.0, **dd)},
    )
    e1 = dftd3(
        structure,
        {**param, "s9": torch.tensor(1.0, **dd)},
    )

    s9 = torch.tensor(0.0, requires_grad=True, **dd)
    energy = dftd3(structure, {**param, "s9": s9}).sum()
    (grad,) = torch.autograd.grad(energy, s9)

    assert pytest.approx(e0.sum().item(), abs=tol) == energy.item()
    assert grad != 0.0
    assert pytest.approx((e1 - e0).sum().item(), abs=tol) == grad.item()


@pytest.mark.grad
@pytest.mark.parametrize("source", [("mb16_43", "SiH4"), ("mb16_43", "01")])
def test_s9_zero_functorch(source: tuple[str, str]) -> None:
    """`jacrev` at ``s9 = 0`` and `vmap` over several `s9` values."""
    numbers, positions, param = _s9_setup(source)
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    def energy(s9: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=numbers, positions=positions), {**param, "s9": s9}
        ).sum()

    s9 = torch.tensor([0.0, 0.5, 1.0], **dd)
    assert vmap_matches_loop(energy, s9, atol=tol)

    grad = torch.func.jacrev(energy)(s9[0])
    e3 = energy(s9[2]) - energy(s9[0])
    assert pytest.approx(e3.item(), abs=tol) == grad.item()
