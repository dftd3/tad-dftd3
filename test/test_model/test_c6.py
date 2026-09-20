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

import pytest
import torch
from tad_mctc._version import __tversion__
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import model, ncoord, reference

from ..conftest import DEVICE, FAST_MODE
from .samples import samples

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]

tol = 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)
    weights = sample["weights"].to(**dd)
    refc6 = sample["c6"].to(**dd)

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == c6.cpu()
    assert pytest.approx(c6.cpu(), abs=tol, rel=tol) == c6.mT.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", ["SiH4"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample1, sample2 = (
        samples[name1],
        samples[name2],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    ref = reference.Reference(**dd)
    weights = pack(
        (
            sample1["weights"].to(**dd),
            sample2["weights"].to(**dd),
        )
    )
    refc6 = pack(
        (
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        )
    )

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == c6.cpu()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
def test_vmap() -> None:
    """
    `numbers` is genuinely batched here (different molecules), so
    `atomic_c6` must fall back to its dense evaluation internally instead
    of calling the illegal-under-`vmap` `numbers.unique()`.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample1, sample2 = samples["SiH4"], samples["PbH4-BiH3"]
    numbers = pack(
        (sample1["numbers"].to(DEVICE), sample2["numbers"].to(DEVICE))
    )
    weights = pack((sample1["weights"].to(**dd), sample2["weights"].to(**dd)))
    ref = reference.Reference(**dd)

    def f(nums: Tensor, ws: Tensor) -> Tensor:
        return model.atomic_c6(nums, ws, ref)

    batched = torch.func.vmap(f, in_dims=(0, 0))(numbers, weights)

    for i, sample in enumerate((sample1, sample2)):
        refc6 = sample["c6"].to(**dd)
        nat = refc6.shape[-1]
        assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == (
            batched[i, :nat, :nat].cpu()
        )


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
def test_jacrev() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)
    weights = sample["weights"].to(**dd)

    def f(ws: Tensor) -> Tensor:
        return model.atomic_c6(numbers, ws, ref)

    jac = torch.func.jacrev(f)(weights)
    nat, nref = weights.shape
    assert jac.shape == (nat, nat, nat, nref)


@pytest.mark.skipif(__tversion__ < (2, 1, 0), reason="Requires PyTorch>=2.1.0")
@pytest.mark.skip(
    reason=(
        "tad-mctc==0.8.0 (currently pinned) has an `is_compiling` check in "
        "`tad_mctc.math.einsum` that under-reports on newer PyTorch, so "
        "`_atomic_c6_safe` falls through to `opt_einsum.contract`, which "
        "Dynamo cannot trace (`threading.get_ident()`). Re-enable once "
        "tad-mctc is updated/pinned to a release with the fixed check."
    )
)
def test_compile() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)
    weights = sample["weights"].to(**dd)
    refc6 = sample["c6"].to(**dd)

    compiled = torch.compile(model.atomic_c6, fullgraph=True)
    c6 = compiled(numbers, weights, ref)

    assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == c6.cpu()


###############################################################################


def gradchecker(
    dtype: torch.dtype,
    name: str,
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    ref = reference.Reference(**dd)
    cn = ncoord.cn_d3(numbers, positions)
    w = model.weight_references(numbers, cn, ref)

    # variables to be differentiated
    w = w.detach().clone().requires_grad_(True)

    def func(weights: Tensor) -> Tensor:
        return model.atomic_c6(numbers, weights, ref)

    return func, w


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["LiH"] + sample_list)
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
