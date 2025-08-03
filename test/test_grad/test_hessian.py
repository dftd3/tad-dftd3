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
Testing dispersion Hessian (autodiff).
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import hess_fn_rev, hessian
from tad_mctc.batch import pack
from tad_mctc.convert import reshape_fortran

from tad_dftd3 import dftd3
from tad_dftd3.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

sample_list = ["LiH", "SiH4", "PbH4-BiH3", "MB16_43_01"]

tol = 1e-8


def test_fail() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"]
    param = {"a1": numbers}

    # differentiable variable is not a tensor
    with pytest.raises(ValueError):
        hessian(dftd3, (numbers, positions, param), argnums=2)


def test_zeros() -> None:
    d = torch.randn(2, 3, requires_grad=True)

    def dummy(x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    hess = hessian(dummy, (d,), argnums=0)
    zeros = torch.zeros([*d.shape, *d.shape])
    assert pytest.approx(zeros.cpu()) == hess.detach().cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    hess = hessian(dftd3, (numbers, positions, param), argnums=1)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()

    positions.detach_()

    # Test with closure over non-tensor argument

    def _energy(numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Closure over non-tensor argument `param` for `dftd3` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd3(numbers, positions, param).sum(-1)

    pos = positions.clone().requires_grad_(True)
    hess = hess_fn_rev(_energy, argnums=1)(numbers, pos)
    assert isinstance(hess, Tensor)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_v2(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    def _energy(numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Closure over non-tensor argument `param` for `dftd3` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd3(numbers, positions, param).sum(-1)

    pos = positions.clone().requires_grad_(True)
    hess = hess_fn_rev(_energy, argnums=1)(numbers, pos)
    assert isinstance(hess, Tensor)

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()

    positions.detach_()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_batch(
    dtype: torch.dtype, name1: str, name2: str, chunk_size: int | None
) -> None:
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

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref = pack(
        [
            reshape_fortran(
                sample1["hessian"].to(**dd),
                torch.Size(2 * (sample1["numbers"].shape[-1], 3)),
            ),
            reshape_fortran(
                sample2["hessian"].to(**dd),
                torch.Size(2 * (sample2["numbers"].shape[-1], 3)),
            ),
        ]
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    def _energy(numbers: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Closure over non-tensor argument `param` for `dftd3` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd3(numbers, positions, param, chunk_size=chunk_size).sum(-1)

    hess_fn = hess_fn_rev(_energy, argnums=1)
    hess_fn_batch = torch.func.vmap(hess_fn, in_dims=(0, 0))

    hess = hess_fn_batch(numbers, positions)
    assert isinstance(hess, Tensor)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()

    positions.detach_()
