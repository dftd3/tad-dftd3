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
from tad_mctc.typing import DD

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import param_on
from ..conftest import DEVICE
from ..samples import mols

cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-10

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _molecule(name: str) -> Structure:
    sample = mols[name]
    return Structure(
        numbers=sample["numbers"].to(DEVICE),
        positions=sample["positions"].to(**DD64),
    )


@pytest.mark.parametrize("name", ["PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"])
def test_matches_dense(name: str) -> None:
    structure = _molecule(name)
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_batch_matches_dense() -> None:
    structure = pack_structures(
        [_molecule("PbH4-BiH3"), _molecule("C6H5I-CH3SH")]
    )
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


@pytest.mark.parametrize("skin", [0.0, 2.0])
def test_wide_list_matches_dense(skin: float) -> None:
    structure = _molecule("C6H5I-CH3SH")
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
    structure = _molecule("C6H5I-CH3SH")
    p = param_on(DD64)

    def grad(sparse: bool) -> torch.Tensor:
        pos = structure.positions.clone().requires_grad_(True)
        s = structure.replace(positions=pos)
        energy = dftd3(s, p, cutoff=cutoff, sparse=sparse).sum()
        return torch.autograd.grad(energy, pos)[0]

    assert pytest.approx(grad(False).cpu(), abs=1e-9) == grad(True).cpu()


def test_mixed_dense_and_sparse() -> None:
    structure = _molecule("C6H5I-CH3SH")
    p = param_on(DD64)
    (nbl_cn,) = build_neighborlists(structure, (cutoff.cn,))

    dense = dftd3(structure, p, cutoff=cutoff)
    mixed = dftd3(structure, p, cutoff=cutoff, nbl_cn=nbl_cn)
    assert pytest.approx(dense.cpu(), abs=tol) == mixed.cpu()
