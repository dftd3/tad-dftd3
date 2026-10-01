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
Testing composition of `torch.func` transforms (`vmap`, `jacrev`, `jacfwd`)
with the full `dftd3` energy.

Batched results are compared to the same quantity evaluated sample by sample.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import dftd3

from ..conftest import DEVICE
from .samples import samples

tol = 1e-8


names = [("LiH", "SiH4"), ("LiH", "PbH4-BiH3")]


def setup(
    dtype: torch.dtype, names: tuple[str, str], s9: float
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor], dict[str, Tensor]]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    nums = [samples[n]["numbers"].to(DEVICE) for n in names]
    pos = [samples[n]["positions"].to(**dd) for n in names]

    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "s9": torch.tensor(s9, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }
    return pack(nums), pack(pos), nums, pos, param


def per_sample(
    fn: Callable[[Tensor, Tensor], Tensor],
    nums: list[Tensor],
    pos: list[Tensor],
) -> Tensor:
    """Evaluate `fn` for each sample separately and pad to a common shape."""
    return pack([fn(n, p) for n, p in zip(nums, pos)])


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_vmap_energy(names: tuple[str, str], s9: float) -> None:
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(n, p, param)

    ref = per_sample(energy, nums, pos)
    out = torch.func.vmap(energy, in_dims=(0, 0))(numbers, positions)

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_vmap_jac(names: tuple[str, str], s9: float, jac: str) -> None:
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    jacfn = getattr(torch.func, jac)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(n, p, param).sum(-1)

    grad = jacfn(energy, argnums=1)
    ref = per_sample(grad, nums, pos)
    out = torch.func.vmap(grad, in_dims=(0, 0))(numbers, positions)

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_jacrev_jacfwd_single(names: tuple[str, str], s9: float) -> None:
    """Forward and reverse mode agree on a single geometry."""
    _, _, nums, pos, param = setup(torch.double, names, s9)

    def energy(p: Tensor) -> Tensor:
        return dftd3(nums[1], p, param).sum(-1)

    rev = torch.func.jacrev(energy)(pos[1])
    fwd = torch.func.jacfwd(energy)(pos[1])

    assert rev.shape == pos[1].shape
    assert pytest.approx(rev.cpu(), abs=tol) == fwd.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_jac_of_vmap(names: tuple[str, str], s9: float, jac: str) -> None:
    """`jac(vmap(...))` over the whole batch (outer derivative)."""
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    jacfn = getattr(torch.func, jac)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(n, p, param).sum(-1)

    def total(p: Tensor) -> Tensor:
        return torch.func.vmap(energy, in_dims=(0, 0))(numbers, p).sum()

    ref = per_sample(torch.func.jacrev(energy, argnums=1), nums, pos)
    out = jacfn(total)(positions)

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
@pytest.mark.parametrize("outer", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("inner", ["jacrev", "jacfwd"])
def test_vmap_hessian(
    names: tuple[str, str], s9: float, outer: str, inner: str
) -> None:
    """Second derivatives under `vmap` for all forward/reverse mixes."""
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(n, p, param).sum(-1)

    def hess(f: Callable[..., Tensor], a: str, b: str) -> Callable[..., Tensor]:
        return getattr(torch.func, a)(
            getattr(torch.func, b)(f, argnums=1), argnums=1
        )

    # `jacrev(jacrev)` of the unbatched energy is the reference
    ref = per_sample(hess(energy, "jacrev", "jacrev"), nums, pos)
    out = torch.func.vmap(hess(energy, outer, inner), in_dims=(0, 0))(
        numbers, positions
    )

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
