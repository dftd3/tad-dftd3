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
from tad_mctc import Structure
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.typing import DD, Callable, Tensor

from tad_dftd3 import data, dftd3
from tad_dftd3.cutoff import Cutoff

from ..conftest import DEVICE
from ..utils import load_sample

tol = 1e-8


names: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("mb16_43", "LiH"), ("mb16_43", "SiH4")),
    (("mb16_43", "LiH"), ("heavy28", "pbh4_bih3")),
]


def setup(
    dtype: torch.dtype,
    names: tuple[tuple[str, str], tuple[str, str]],
    s9: float,
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor], dict[str, Tensor]]:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    nums = [load_sample(*src, dd)[0] for src in names]
    pos = [load_sample(*src, dd)[1] for src in names]

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
def test_vmap_energy(
    names: tuple[tuple[str, str], tuple[str, str]], s9: float
) -> None:
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=n, positions=p), param)

    ref = per_sample(energy, nums, pos)
    out = torch.func.vmap(energy, in_dims=(0, 0))(numbers, positions)

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_vmap_jac(
    names: tuple[tuple[str, str], tuple[str, str]], s9: float, jac: str
) -> None:
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    jacfn = getattr(torch.func, jac)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=n, positions=p), param).sum(-1)

    grad = jacfn(energy, argnums=1)
    ref = per_sample(grad, nums, pos)
    out = torch.func.vmap(grad, in_dims=(0, 0))(numbers, positions)

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_jacrev_jacfwd_single(
    names: tuple[tuple[str, str], tuple[str, str]], s9: float
) -> None:
    """Forward and reverse mode agree on a single geometry."""
    _, _, nums, pos, param = setup(torch.double, names, s9)

    def energy(p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=nums[1], positions=p), param).sum(-1)

    rev = torch.func.jacrev(energy)(pos[1])
    fwd = torch.func.jacfwd(energy)(pos[1])

    assert rev.shape == pos[1].shape
    assert pytest.approx(rev.cpu(), abs=tol) == fwd.cpu()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
@pytest.mark.parametrize("jac", ["jacrev", "jacfwd"])
def test_jac_of_vmap(
    names: tuple[tuple[str, str], tuple[str, str]], s9: float, jac: str
) -> None:
    """`jac(vmap(...))` over the whole batch (outer derivative)."""
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    jacfn = getattr(torch.func, jac)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=n, positions=p), param).sum(-1)

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
    names: tuple[tuple[str, str], tuple[str, str]],
    s9: float,
    outer: str,
    inner: str,
) -> None:
    """Second derivatives under `vmap` for all forward/reverse mixes."""
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=n, positions=p), param).sum(-1)

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


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
# Forward mode over the (Z, Z) `rvdw` table pushes ~10⁴ tangents.
@pytest.mark.parametrize(
    "jac, name",
    [
        ("jacrev", "rcov"),
        ("jacrev", "rvdw"),
        ("jacrev", "r4r2"),
        ("jacfwd", "rcov"),
        ("jacfwd", "r4r2"),
    ],
)
def test_vmap_jac_table(
    names: tuple[tuple[str, str], tuple[str, str]],
    s9: float,
    jac: str,
    name: str,
) -> None:
    """
    Per-element gradients of a padded batch, with the table shared by all
    systems. The padding atoms must neither turn the gradients non-finite
    nor send a gradient to the dummy entry 0 of the table.
    """
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    table = {
        "rcov": radii.COV_D3,
        "rvdw": radii.VDW_PAIRWISE,
        "r4r2": data.R4R2,
    }[name](**dd)

    def energy(n: Tensor, p: Tensor, t: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=n, positions=p), param, **{f"{name}_table": t}
        ).sum(-1)

    # `jacrev` of the unbatched energy is the reference
    rev = torch.func.jacrev(energy, argnums=(1, 2))
    ref_pos = per_sample(lambda n, p: rev(n, p, table)[0], nums, pos)
    ref_table = torch.stack([rev(n, p, table)[1] for n, p in zip(nums, pos)])

    grad = getattr(torch.func, jac)(energy, argnums=(1, 2))

    out_pos, out_table = torch.func.vmap(grad, in_dims=(0, 0, None))(
        numbers, positions, table
    )

    assert torch.isfinite(out_pos).all()
    assert torch.isfinite(out_table).all()

    assert out_table.shape == (numbers.shape[0], *table.shape)
    assert pytest.approx(ref_table.cpu(), abs=tol) == out_table.cpu()
    assert pytest.approx(ref_pos.cpu(), abs=tol) == out_pos.cpu()

    # nothing reaches the dummy entry or the padding atoms
    assert (out_table[:, 0] == 0).all()
    if name == "rvdw":
        assert (out_table[:, :, 0] == 0).all()
    assert (out_pos[numbers == 0] == 0).all()


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_vmap_jacrev_changed_cutoff(
    names: tuple[tuple[str, str], tuple[str, str]], s9: float
) -> None:
    """
    `vmap(jacrev)` with cutoffs short enough to cut pairs and triples of
    these molecules, so the cutoffs reach the coordination number and both
    dispersion terms in the traced graph.
    """
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    cutoff = Cutoff(cn=4.0, disp2=6.0, disp3=6.0)

    def energy(n: Tensor, p: Tensor, c: Cutoff | None) -> Tensor:
        return dftd3(Structure(numbers=n, positions=p), param, cutoff=c).sum(-1)

    def grad(c: Cutoff | None) -> Callable[[Tensor, Tensor], Tensor]:
        return torch.func.jacrev(lambda n, p: energy(n, p, c), argnums=1)

    ref = per_sample(grad(cutoff), nums, pos)
    out = torch.func.vmap(grad(cutoff), in_dims=(0, 0))(numbers, positions)

    assert out.shape == ref.shape
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()

    # only meaningful if the cutoffs change the result
    default = torch.func.vmap(grad(None), in_dims=(0, 0))(numbers, positions)
    assert not torch.allclose(out, default, atol=1e-10, rtol=0)


@pytest.mark.parametrize("names", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_vmap_structure(
    names: tuple[tuple[str, str], tuple[str, str]], s9: float
) -> None:
    """
    `vmap` directly over a padded batch of `Structure`s, and derivatives by
    replacing the positions in the structure.
    """
    numbers, positions, nums, pos, param = setup(torch.double, names, s9)
    batch = Structure(numbers=numbers, positions=positions)

    def energy(structure: Structure) -> Tensor:
        return dftd3(structure, param).sum(-1)

    def grad(structure: Structure) -> Tensor:
        def from_positions(p: Tensor) -> Tensor:
            return energy(structure.replace(positions=p))

        return torch.func.jacrev(from_positions)(structure.positions)

    def ref_energy(n: Tensor, p: Tensor) -> Tensor:
        return energy(Structure(numbers=n, positions=p))

    ref = torch.stack([ref_energy(n, p) for n, p in zip(nums, pos)])
    out = torch.func.vmap(energy)(batch)
    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()

    ref_grad = per_sample(torch.func.jacrev(ref_energy, argnums=1), nums, pos)
    out_grad = torch.func.vmap(grad)(batch)
    assert pytest.approx(ref_grad.cpu(), abs=tol) == out_grad.cpu()
