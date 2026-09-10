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
Test C6 coefficients.
"""

from __future__ import annotations

from typing import Protocol

import pytest
import torch
from tad_mctc._version import __tversion__
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.data.molecules import mols
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import model, ncoord, reference

from ..conftest import DEVICE, FAST_MODE
from ..references import reference_c6, reference_weights

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]

tol = 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    numbers = mols[name]["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)

    # weights and c6 both come from s-dftd3's Fortran library (see
    # test/references). Feeding in the reference weights isolates this test
    # to atomic_c6, independent of weight_references.
    weights = reference_weights(name, dd)
    refc6 = reference_c6(name, dd)

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == c6.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    numbers = pack(
        (
            mols[name1]["numbers"].to(DEVICE),
            mols[name2]["numbers"].to(DEVICE),
        )
    )
    ref = reference.Reference(**dd)

    # s-dftd3 has no notion of a batch of independent molecules; the
    # reference for each molecule was computed on its own and is packed
    # here the same way the inputs above were packed.
    weights = pack((reference_weights(name1, dd), reference_weights(name2, dd)))
    refc6 = pack((reference_c6(name1, dd), reference_c6(name2, dd)))

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == c6.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [100, 200, 500])
@pytest.mark.parametrize("chunk_size", [10, 100])
def test_chunked(dtype: torch.dtype, size: int, chunk_size: int) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    ref = reference.Reference(**dd)
    numbers = torch.randint(1, 86, (size,), device=DEVICE)
    positions = torch.rand((size, 3), **dd) * 10

    cn = ncoord.cn_d3(numbers, positions)
    weights = model.weight_references(numbers, cn, ref)

    c6 = model.atomic_c6(numbers, weights, ref)
    c6_chunked = model.atomic_c6(numbers, weights, ref, chunk_size=chunk_size)

    assert c6.dtype == c6_chunked.dtype == dtype
    assert pytest.approx(c6.cpu(), abs=tol, rel=tol) == c6_chunked.cpu()


@pytest.mark.skipif(__tversion__ < (2, 1, 0), reason="Requires PyTorch>=2.1.0")
def test_fail() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}
    size = 10
    nbatch = 2

    ref = reference.Reference(**dd)
    numbers = torch.randint(1, 86, (nbatch, size), device=DEVICE)
    positions = torch.rand((nbatch, size, 3), **dd) * 10

    cn = ncoord.cn_d3(numbers, positions)
    weights = model.weight_references(numbers, cn, ref)

    def _c6(nums: Tensor, ws: Tensor) -> Tensor:
        return model.atomic_c6(nums, ws, ref)

    jac = torch.func.jacrev(_c6, argnums=1)

    # Correct batch dimensions
    with pytest.raises(ValueError) as excinfo:
        vjac = torch.func.vmap(jac, in_dims=(0, None))
        _ = vjac(numbers, weights.moveaxis(1, 0))

    assert "Batch size mismatch" in str(excinfo.value)
    assert "weights" in str(excinfo.value)

    # Correct batch dimensions
    with pytest.raises(ValueError) as excinfo:
        vjac = torch.func.vmap(jac, in_dims=(None, 0))
        _ = vjac(numbers.moveaxis(1, 0), weights)

    assert "Batch size mismatch" in str(excinfo.value)
    assert "numbers" in str(excinfo.value)

    # Internal vmap errors

    # Batch dimensions is always 0 for numbers and weights
    with pytest.raises(ValueError) as excinfo:
        vjac = torch.func.vmap(jac, in_dims=(1, 0))
        _ = vjac(numbers, weights)

    # Batch dimensions is always 0 for numbers and weights
    with pytest.raises(ValueError):
        vjac = torch.func.vmap(jac, in_dims=(0, 1))
        _ = vjac(numbers, weights)


###############################################################################


class C6Func(Protocol):
    """
    Type annotation for a function that calculates C6 coefficients.
    """

    def __call__(
        self,
        numbers: Tensor,
        weights: Tensor,
        ref: reference.Reference,
        chunk_size: int | None = None,
    ) -> Tensor: ...


def gradchecker(
    dtype: torch.dtype,
    name: str,
    f: C6Func,
    chunk_size: int | None = None,
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # Only numbers/positions are needed here -- cn and weights are computed
    # live below, so "LiH" (which has no reference fixture data) works too.
    mol = mols[name]
    numbers = mol["numbers"].to(DEVICE)
    positions = mol["positions"].to(**dd)

    ref = reference.Reference(**dd)
    cn = ncoord.cn_d3(numbers, positions)
    w = model.weight_references(numbers, cn, ref)

    # variables to be differentiated
    w = w.detach().clone().requires_grad_(True)

    def func(weights: Tensor) -> Tensor:
        if chunk_size is None:
            return f(numbers, weights, ref)
        return f(numbers, weights, ref, chunk_size)

    return func, w


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LiH"] + sample_list)
@pytest.mark.parametrize(
    "f, chunk_size",
    [
        (model.c6._atomic_c6_full, None),
        (model.c6._atomic_c6_chunked, 2),
        (model.atomic_c6, None),
        (model.atomic_c6, 2),
        (model.c6.AtomicC6_V1.apply, None),
        (model.c6.AtomicC6_V1.apply, 2),
    ],
)
def test_gradcheck(
    dtype: torch.dtype, name: str, f: C6Func, chunk_size: int | None
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, f, chunk_size)
    assert dgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize(
    "f, chunk_size",
    [
        (model.c6._atomic_c6_full, None),
        (model.c6._atomic_c6_chunked, 2),
        (model.atomic_c6, None),
        (model.atomic_c6, 2),
        (model.c6.AtomicC6_V1.apply, None),
        (model.c6.AtomicC6_V1.apply, 2),
    ],
)
def test_gradgradcheck(
    dtype: torch.dtype, name: str, f: C6Func, chunk_size: int | None
) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name, f, chunk_size=chunk_size)
    assert dgradgradcheck(func, diffvars, atol=tol, fast_mode=FAST_MODE)
