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
"""
Sparse molecular quadrant: :func:`tad_dftd3.dftd3` with neighbour lists,
for molecules and batches of them, against the dense evaluation.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import build_neighborlists
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD

from tad_dftd3 import dftd3, disp
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.model import AtomicC6

from ..cells import param_on
from ..conftest import DEVICE, compile_test
from ..utils import load_structure

cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _molecule(source: tuple[str, str]) -> Structure:
    return load_structure(*source, DD64)


@pytest.mark.parametrize(
    "source",
    [("heavy28", "pbh4_bih3"), ("other", "C6H5I-CH3SH"), ("mb16_43", "01")],
)
def test_matches_dense(source: tuple[str, str]) -> None:
    structure = _molecule(source)
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_batch_matches_dense() -> None:
    structure = pack_structures(
        [
            _molecule(("heavy28", "pbh4_bih3")),
            _molecule(("other", "C6H5I-CH3SH")),
        ]
    )
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


@pytest.mark.parametrize("skin", [0.0, 2.0])
def test_wide_list_matches_dense(skin: float) -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = param_on(DD64)

    # built at larger cutoffs (and with a skin) than used
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn + 5.0, cutoff.disp2 + 5.0), skin=skin
    )

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(
        structure, p, cutoff=cutoff, nbl_cn=nbl_cn, nbl_disp2=nbl_disp2
    )
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_gradient_matches_dense() -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = param_on(DD64)

    def grad(sparse: bool) -> torch.Tensor:
        pos = structure.positions.clone().requires_grad_(True)
        s = structure.replace(positions=pos)
        energy = dftd3(s, p, cutoff=cutoff, sparse=sparse).sum()
        return torch.autograd.grad(energy, pos)[0]

    assert pytest.approx(grad(False).cpu(), abs=1e-12) == grad(True).cpu()


def test_mixed_dense_and_sparse() -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = param_on(DD64)
    (nbl_cn,) = build_neighborlists(structure, (cutoff.cn,))

    dense = dftd3(structure, p, cutoff=cutoff)
    mixed = dftd3(structure, p, cutoff=cutoff, nbl_cn=nbl_cn)
    assert pytest.approx(dense.cpu(), abs=tol) == mixed.cpu()


def test_jacrev_matches_dense() -> None:
    """Over the lists, C6 is factored per atom (see `AtomicC6`)."""
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = param_on(DD64)
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2)
    )

    def energy(pos: torch.Tensor, sparse: bool) -> torch.Tensor:
        moved = structure.replace(positions=pos)
        if not sparse:
            return dftd3(moved, p, cutoff=cutoff).sum()
        return dftd3(
            moved, p, cutoff=cutoff, nbl_cn=nbl_cn, nbl_disp2=nbl_disp2
        ).sum()

    pos = structure.positions
    dense = torch.func.jacrev(energy)(pos, False)
    sparse = torch.func.jacrev(energy)(pos, True)
    assert pytest.approx(dense.cpu(), abs=1e-12) == sparse.cpu()


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compile_factored_c6() -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = param_on(DD64)
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2)
    )
    model = disp.D3Model(cutoff=cutoff)
    c6 = model.factored_c6(structure, nbl_cn)

    def energy(c6: AtomicC6) -> torch.Tensor:
        return disp.dispersion2(
            structure, p, c6, pairs=nbl_disp2, cutoff=cutoff
        )

    compiled = compile_test(energy, fullgraph=True)
    dense = disp.dispersion2(structure, p, c6.dense(), cutoff=cutoff)
    assert pytest.approx(dense.cpu(), abs=tol) == compiled(c6).cpu()
