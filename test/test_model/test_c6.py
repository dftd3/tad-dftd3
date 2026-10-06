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
from tad_mctc.convert import symbol_to_number
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import model, ncoord, reference
from tad_dftd3.model import c6 as c6_module

from ..conftest import DEVICE, FAST_MODE, compile_test
from ..references import reference_c6, reference_weights
from ..utils import approx_ref, load_sample

sample_list: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),
    ("heavy28", "pbh4_bih3"),
    ("other", "C6H5I-CH3SH"),
    ("mb16_43", "01"),
]

tol = 1e-8

DD_CPU64: DD = {"device": torch.device("cpu"), "dtype": torch.double}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source", sample_list)
def test_single(dtype: torch.dtype, source: tuple[str, str]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = load_sample(*source, dd)[0]
    ref = reference.Reference.load(**dd)

    # weights and C6 both come from s-dftd3's Fortran library (see
    # test/references); feeding in its weights isolates this test to atomic_c6
    weights = reference_weights(*source, dd)
    refc6 = reference_c6(*source, dd)

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert approx_ref(refc6.cpu(), dtype) == c6.cpu()
    assert approx_ref(c6.cpu(), dtype) == c6.mT.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source1", [("mb16_43", "SiH4")])
@pytest.mark.parametrize("source2", sample_list)
def test_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        (
            load_sample(*source1, dd)[0],
            load_sample(*source2, dd)[0],
        )
    )
    ref = reference.Reference.load(**dd)

    # s-dftd3 has no notion of a batch of independent molecules; each
    # reference is packed like the inputs above
    weights = pack(
        (reference_weights(*source1, dd), reference_weights(*source2, dd))
    )
    refc6 = pack((reference_c6(*source1, dd), reference_c6(*source2, dd)))

    c6 = model.atomic_c6(numbers, weights, ref)

    assert c6.dtype == dtype
    assert approx_ref(refc6.cpu(), dtype) == c6.cpu()


def test_vmap() -> None:
    """
    `numbers` is genuinely batched here (different molecules), so
    `atomic_c6` must fall back to its dense evaluation internally instead
    of calling the illegal-under-`vmap` `numbers.unique()`.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sources = [("mb16_43", "SiH4"), ("heavy28", "pbh4_bih3")]
    numbers = pack(tuple(load_sample(*src, dd)[0] for src in sources))
    weights = pack(tuple(reference_weights(*src, dd) for src in sources))
    ref = reference.Reference.load(**dd)

    def f(nums: Tensor, ws: Tensor) -> Tensor:
        return model.atomic_c6(nums, ws, ref)

    batched = torch.func.vmap(f, in_dims=(0, 0))(numbers, weights)

    for i, src in enumerate(sources):
        refc6 = reference_c6(*src, dd)
        nat = refc6.shape[-1]
        assert approx_ref(refc6.cpu(), torch.double) == (
            batched[i, :nat, :nat].cpu()
        )


def test_jacrev() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = load_sample("mb16_43", "SiH4", dd)[0]
    ref = reference.Reference.load(**dd)
    weights = reference_weights("mb16_43", "SiH4", dd)

    def f(ws: Tensor) -> Tensor:
        return model.atomic_c6(numbers, ws, ref)

    jac = torch.func.jacrev(f)(weights)
    nat, nref = weights.shape
    assert jac.shape == (nat, nat, nat, nref)


@requires_compile
def test_compile() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = load_sample("mb16_43", "SiH4", dd)[0]
    ref = reference.Reference.load(**dd)
    weights = reference_weights("mb16_43", "SiH4", dd)
    refc6 = reference_c6("mb16_43", "SiH4", dd)

    compiled = compile_test(model.atomic_c6, fullgraph=True)
    c6 = compiled(numbers, weights, ref)

    assert approx_ref(refc6.cpu(), torch.double) == c6.cpu()


###############################################################################


def _all_pairs(nat: int, nbatch: int = 1) -> tuple[Tensor, Tensor]:
    """Every pair of atoms of each system, in the flat numbering of a list."""
    i, j = torch.meshgrid(
        torch.arange(nat, device=DEVICE),
        torch.arange(nat, device=DEVICE),
        indexing="ij",
    )
    offset = (torch.arange(nbatch, device=DEVICE) * nat)[:, None]
    return (
        (i.reshape(1, -1) + offset).reshape(-1),
        (j.reshape(1, -1) + offset).reshape(-1),
    )


def test_factored_pairs_match_dense() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sources = [("mb16_43", "SiH4"), ("heavy28", "pbh4_bih3")]
    numbers = pack(tuple(load_sample(*src, dd)[0] for src in sources))
    weights = pack(tuple(reference_weights(*src, dd) for src in sources))
    ref = reference.Reference.load(**dd)

    c6 = model.factored_c6(numbers, weights, ref)
    dense = model.atomic_c6(numbers, weights, ref)

    idx_i, idx_j = _all_pairs(numbers.shape[-1], numbers.shape[0])
    pairs = c6.pair(idx_i, idx_j).reshape(dense.shape)
    assert pytest.approx(dense.cpu(), abs=1e-12) == pairs.cpu()


def test_factored_vmap() -> None:
    """Batched `numbers` take the factors over all elements."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    sources = [("mb16_43", "SiH4"), ("heavy28", "pbh4_bih3")]
    numbers = pack(tuple(load_sample(*src, dd)[0] for src in sources))
    weights = pack(tuple(reference_weights(*src, dd) for src in sources))
    ref = reference.Reference.load(**dd)
    idx_i, idx_j = _all_pairs(numbers.shape[-1])

    def f(nums: Tensor, ws: Tensor) -> Tensor:
        return model.factored_c6(nums, ws, ref).pair(idx_i, idx_j)

    batched = torch.func.vmap(f)(numbers, weights)
    dense = model.atomic_c6(numbers, weights, ref)
    assert pytest.approx(dense.cpu(), abs=1e-12) == (
        batched.reshape(dense.shape).cpu()
    )


def test_factored_jacrev() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = load_sample("other", "C6H5I-CH3SH", dd)[0]
    ref = reference.Reference.load(**dd)
    weights = reference_weights("other", "C6H5I-CH3SH", dd)
    idx_i, idx_j = _all_pairs(numbers.shape[-1])

    def pairs(ws: Tensor) -> Tensor:
        return model.factored_c6(numbers, ws, ref).pair(idx_i, idx_j)

    def dense(ws: Tensor) -> Tensor:
        return model.atomic_c6(numbers, ws, ref).reshape(-1)

    jac = torch.func.jacrev(pairs)(weights)
    assert pytest.approx(torch.func.jacrev(dense)(weights).cpu()) == jac.cpu()


def _full_width(
    numbers: Tensor, weights: Tensor, ref: reference.Reference
) -> model.AtomicC6:
    """The factors over all 7 references, as before the cut."""
    elements = torch.unique(numbers)
    element = torch.searchsorted(elements, numbers.contiguous())
    return c6_module._factors(numbers, weights, ref, elements, element)


@pytest.mark.parametrize(
    "symbols, width",
    [(["H", "H"], 2), (["O", "H", "H"], 3), (["C", "H", "N", "O"], 5)],
)
def test_reference_width(symbols: list[str], width: int) -> None:
    """The reference axis is cut to the most references of an element."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = symbol_to_number(symbols).to(DEVICE)
    ref = reference.Reference.load(**dd)
    gen = torch.Generator().manual_seed(0)
    weights = torch.rand(numbers.shape[0], 7, generator=gen, **DD_CPU64)
    weights = weights.to(DEVICE) * (ref.cn[numbers] >= 0)

    c6 = model.factored_c6(numbers, weights, ref)
    assert c6.partial.shape[-1] == width and c6.weights.shape[-1] == width

    full = _full_width(numbers, weights, ref)
    assert full.partial.shape[-1] == 7
    assert pytest.approx(full.dense().cpu(), abs=1e-14) == c6.dense().cpu()


def test_reference_width_jacrev() -> None:
    """The weights of the references that were cut get a zero derivative."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = load_sample("other", "C6H5I-CH3SH", dd)[0]
    ref = reference.Reference.load(**dd)
    weights = reference_weights("other", "C6H5I-CH3SH", dd)

    def cut(ws: Tensor) -> Tensor:
        return model.factored_c6(numbers, ws, ref).dense()

    def full(ws: Tensor) -> Tensor:
        return _full_width(numbers, ws, ref).dense()

    jac = torch.func.jacrev(cut)(weights)
    assert pytest.approx(torch.func.jacrev(full)(weights).cpu()) == jac.cpu()


@requires_compile
def test_factored_compile() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers = load_sample("mb16_43", "SiH4", dd)[0]
    ref = reference.Reference.load(**dd)
    weights = reference_weights("mb16_43", "SiH4", dd)
    idx_i, idx_j = _all_pairs(numbers.shape[-1])

    def f(nums: Tensor, ws: Tensor) -> Tensor:
        return model.factored_c6(nums, ws, ref).pair(idx_i, idx_j)

    compiled = compile_test(f, fullgraph=True)
    dense = model.atomic_c6(numbers, weights, ref).reshape(-1)
    assert (
        pytest.approx(dense.cpu(), abs=1e-12)
        == compiled(numbers, weights).cpu()
    )


###############################################################################


def gradchecker(
    dtype: torch.dtype,
    source: tuple[str, str],
) -> tuple[
    Callable[[Tensor], Tensor],  # autograd function
    Tensor,  # differentiable variables
]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = load_sample(*source, dd)[0]
    positions = load_sample(*source, dd)[1]

    ref = reference.Reference.load(**dd)
    cn = ncoord.cn_d3(Structure(numbers=numbers, positions=positions))
    w = model.weight_references(numbers, cn, ref)

    # variables to be differentiated
    w = w.detach().clone().requires_grad_(True)

    def func(weights: Tensor) -> Tensor:
        return model.atomic_c6(numbers, weights, ref)

    return func, w


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", [("mb16_43", "LiH")] + sample_list)
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
