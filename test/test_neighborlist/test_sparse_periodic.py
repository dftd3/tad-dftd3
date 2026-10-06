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
Sparse periodic quadrant: :func:`tad_dftd3.dftd3` with neighbour lists built
for cells, against the dense periodic evaluation (which ``test/test_periodic``
checks against s-dftd3).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import build_neighborlists
from tad_mctc.typing import DD

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import TRICLINIC, cells, param_on, random_cell
from ..conftest import DEVICE

cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}


@pytest.mark.parametrize("name", list(cells))
def test_matches_dense(name: str) -> None:
    structure = cells[name]
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_tiny_cell_self_images() -> None:
    # An atom sees its own images on both sides, all within the cutoff:
    # the stored entry has `i == j` and must count for both directions.
    lattice = torch.eye(3, dtype=torch.double) * 4.0
    structure = random_cell(lattice, 2, DD64)
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_batch_matches_dense() -> None:
    structure = pack_structures(
        [random_cell(TRICLINIC, n, DD64, seed=n) for n in (3, 5)]
    )
    p = param_on(DD64)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


@pytest.mark.parametrize("skin", [0.0, 2.0])
def test_wide_list_matches_dense(skin: float) -> None:
    structure = cells["urea"]
    p = param_on(DD64)

    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn + 5.0, cutoff.disp2 + 5.0), skin=skin
    )

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(
        structure, p, cutoff=cutoff, nbl_cn=nbl_cn, nbl_disp2=nbl_disp2
    )
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_mixed_dense_and_sparse() -> None:
    structure = cells["urea"]
    p = param_on(DD64)
    (nbl_cn,) = build_neighborlists(structure, (cutoff.cn,))

    dense = dftd3(structure, p, cutoff=cutoff)
    mixed = dftd3(structure, p, cutoff=cutoff, nbl_cn=nbl_cn)
    assert pytest.approx(dense.cpu(), abs=tol) == mixed.cpu()


def test_gradient_matches_dense() -> None:
    """Gradients with respect to the positions and the lattice."""
    structure = cells["urea"]
    assert structure.lattice is not None
    p = param_on(DD64)

    def grads(sparse: bool) -> tuple[torch.Tensor, ...]:
        pos = structure.positions.clone().requires_grad_(True)
        assert structure.lattice is not None
        lattice = structure.lattice.clone().requires_grad_(True)
        s = structure.replace(positions=pos, lattice=lattice)
        energy = dftd3(s, p, cutoff=cutoff, sparse=sparse).sum()
        return torch.autograd.grad(energy, (pos, lattice))

    for d, s in zip(grads(False), grads(True)):
        assert pytest.approx(d.cpu(), abs=1e-12) == s.cpu()
