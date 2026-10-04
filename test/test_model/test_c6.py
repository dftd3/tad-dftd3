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
from tad_mctc import Structure
from tad_mctc.autograd import dgradcheck, dgradgradcheck
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import model, ncoord, reference

from ..conftest import DEVICE, FAST_MODE, compile_test, requires_compile
from ..references import reference_c6, reference_weights
from ..samples import mols

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]

tol = 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    numbers = mols[name]["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)

    # weights and C6 both come from s-dftd3's Fortran library (see
    # test/references); feeding in its weights isolates this test to atomic_c6
    weights = reference_weights(name, dd)
    refc6 = reference_c6(name, dd)

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

    numbers = pack(
        (
            mols[name1]["numbers"].to(DEVICE),
            mols[name2]["numbers"].to(DEVICE),
        )
    )
    ref = reference.Reference(**dd)

    # s-dftd3 has no notion of a batch of independent molecules; each
    # reference is packed like the inputs above
    weights = pack((reference_weights(name1, dd), reference_weights(name2, dd)))
    refc6 = pack((reference_c6(name1, dd), reference_c6(name2, dd)))

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == c6.cpu()


def test_vmap() -> None:
    """
    `numbers` is genuinely batched here (different molecules), so
    `atomic_c6` must fall back to its dense evaluation internally instead
    of calling the illegal-under-`vmap` `numbers.unique()`.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    names = ("SiH4", "PbH4-BiH3")
    numbers = pack(tuple(mols[n]["numbers"].to(DEVICE) for n in names))
    weights = pack(tuple(reference_weights(n, dd) for n in names))
    ref = reference.Reference(**dd)

    def f(nums: Tensor, ws: Tensor) -> Tensor:
        return model.atomic_c6(nums, ws, ref)

    batched = torch.func.vmap(f, in_dims=(0, 0))(numbers, weights)

    for i, name in enumerate(names):
        refc6 = reference_c6(name, dd)
        nat = refc6.shape[-1]
        assert pytest.approx(refc6.cpu(), abs=tol, rel=tol) == (
            batched[i, :nat, :nat].cpu()
        )


def test_jacrev() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = mols["SiH4"]["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)
    weights = reference_weights("SiH4", dd)

    def f(ws: Tensor) -> Tensor:
        return model.atomic_c6(numbers, ws, ref)

    jac = torch.func.jacrev(f)(weights)
    nat, nref = weights.shape
    assert jac.shape == (nat, nat, nat, nref)


@requires_compile
def test_compile() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = mols["SiH4"]["numbers"].to(DEVICE)
    ref = reference.Reference(**dd)
    weights = reference_weights("SiH4", dd)
    refc6 = reference_c6("SiH4", dd)

    compiled = compile_test(model.atomic_c6, fullgraph=True)
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

    numbers = mols[name]["numbers"].to(DEVICE)
    positions = mols[name]["positions"].to(**dd)

    ref = reference.Reference(**dd)
    cn = ncoord.cn_d3(Structure(numbers=numbers, positions=positions))
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
