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
The dampings ported from dftd in the D3 energy.

There are no reference values for them yet: dftd is work in progress, and
s-dftd3 does not have them. Every one of them goes through `dftd3`: the
sparse evaluation agrees with the dense one for a molecule, a padded batch
and a cell, the gradient of a padded batch is finite, and the energy runs
under the transforms of ``torch``. The Koide damping is not tested yet.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import Cutoff, Damping, DampingParam, dftd3
from tad_dftd3.damping import damping_from_name

from ..cells import cells
from ..conftest import DEVICE, compile_test
from ..utils import load_structure

DD64: DD = {"device": DEVICE, "dtype": torch.double}

cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=10.0)


def _param() -> DampingParam:
    """One set for all dampings, every value a tensor."""
    values = {
        "s6": 1.0,
        "s8": 0.8,
        "s9": 1.0,
        "a1": 0.45,
        "a2": 4.2,
        "a3": 0.9,
        "a4": 1.1,
        "rs6": 1.2,
        "rs8": 1.0,
        "rs9": 1.1,
        "alp": 16.0,
    }
    tensors: dict[str, Any] = {
        k: torch.tensor(v, **DD64) for k, v in values.items()
    }
    return DampingParam(**tensors)


# The new two-body dampings alone, the new three-body dampings after the
# rational two-body damping, with the D3 damping each replaces.
CASES = {
    "screened-2b": (
        damping_from_name("screened", False),
        damping_from_name("rational", False),
    ),
    **{
        f"{name}-3b": (
            damping_from_name("rational", name),
            damping_from_name("rational", "zero_d3"),
        )
        for name in ("rational", "screened", "zero_d4", "zero_product")
    },
}
NAMES = list(CASES)


def _case(name: str) -> tuple[Damping, DampingParam]:
    return CASES[name][0], _param()


def molecule() -> Structure:
    return load_structure("mb16_43", "SiH4", DD64)


def batch() -> Structure:
    """Two molecules of different size, so the smaller one is padded."""
    return pack_structures(
        [
            load_structure("mb16_43", "LiH", DD64),
            load_structure("mb16_43", "SiH4", DD64),
        ]
    )


def cell() -> Structure:
    return cells["urea"].to(DEVICE)


@pytest.mark.parametrize("name", NAMES)
def test_differs_from_d3(name: str) -> None:
    """The damping changes the energy, so the tests below can see it."""
    damping, param = _case(name)
    structure = molecule()

    out = dftd3(structure, param, cutoff=cutoff, damping=damping)
    ref = dftd3(structure, param, cutoff=cutoff, damping=CASES[name][1])
    assert not torch.allclose(out, ref, atol=1e-10, rtol=0)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["molecule", "batch", "cell"])
def test_sparse_matches_dense(name: str, kind: str) -> None:
    """The neighbour-list evaluation of every term agrees with the dense."""
    damping, param = _case(name)
    structure = {"molecule": molecule, "batch": batch, "cell": cell}[kind]()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, param, cutoff=cutoff, damping=damping)
    sparse = dftd3(
        structure,
        param,
        cutoff=cutoff,
        damping=damping,
        sparse=True,
        nbl_disp3=nbl,
    )
    assert pytest.approx(dense.cpu(), abs=1e-12, rel=0) == sparse.cpu()


@pytest.mark.parametrize("name", NAMES)
def test_padded_gradient_is_finite(name: str) -> None:
    """Padding atoms and the masked pairs and triples give no NaN."""
    damping, param = _case(name)
    structure = batch()
    positions = structure.positions.clone().requires_grad_(True)
    energy = dftd3(
        structure.replace(positions=positions),
        param,
        cutoff=cutoff,
        damping=damping,
    )
    (grad,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.isfinite(energy).all() and torch.isfinite(grad).all()
    assert (grad[structure.numbers == 0] == 0).all()


@pytest.mark.parametrize("name", NAMES)
def test_vmap(name: str) -> None:
    damping, param = _case(name)
    structure = batch()

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=n, positions=p),
            param,
            cutoff=cutoff,
            damping=damping,
        )

    batched = torch.func.vmap(energy)(structure.numbers, structure.positions)
    looped = dftd3(structure, param, cutoff=cutoff, damping=damping)
    assert pytest.approx(looped.cpu(), abs=1e-10, rel=0) == batched.cpu()


@pytest.mark.parametrize("name", NAMES)
def test_jacrev_jacfwd(name: str) -> None:
    damping, param = _case(name)
    structure = molecule()

    def energy(p: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=structure.numbers, positions=p),
            param,
            cutoff=cutoff,
            damping=damping,
        ).sum()

    rev = torch.func.jacrev(energy)(structure.positions)
    fwd = torch.func.jacfwd(energy)(structure.positions)
    assert pytest.approx(rev.cpu(), abs=1e-10, rel=0) == fwd.cpu()


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.parametrize("name", NAMES)
def test_compile_fullgraph(name: str) -> None:
    damping, param = _case(name)
    structure = molecule()

    def energy(p: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=structure.numbers, positions=p),
            param,
            cutoff=cutoff,
            damping=damping,
        )

    ref = energy(structure.positions)
    out = compile_test(energy, fullgraph=True)(structure.positions)
    assert pytest.approx(ref.cpu(), abs=1e-10, rel=0) == out.cpu()


@pytest.mark.parametrize(
    "name, needed",
    [
        ("screened-2b", "a3"),
        ("screened-2b", "a4"),
        ("rational-3b", "s9"),
        ("screened-3b", "a3"),
        ("zero_product-3b", "rs9"),
        ("zero_product-3b", "alp"),
        ("zero_d4-3b", "alp"),
    ],
)
def test_missing_parameters(name: str, needed: str) -> None:
    """
    The parameters a damping requires are not made up, e.g. `alp` and `rs9`,
    whose D3 defaults are not dftd's.
    """
    damping, param = _case(name)
    with pytest.raises(ValueError, match=f"requires.*'{needed}'"):
        dftd3(molecule(), param.replace(**{needed: None}), damping=damping)


def test_radius_defaults() -> None:
    """`a1` and `a2` default to 1 and 0, the damping radius as it is."""
    damping, param = _case("screened-2b")  # no D3 damping, which needs them
    unset = param.replace(a1=None, a2=None)
    explicit = param.replace(a1=1.0, a2=0.0)
    assert torch.equal(
        dftd3(molecule(), unset, cutoff=cutoff, damping=damping),
        dftd3(molecule(), explicit, cutoff=cutoff, damping=damping),
    )
