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
Sparse three-body (ATM) term, summed over the triples of a neighbour list,
against the dense all-triples evaluation. Molecules only: a cell has no
three-body term.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.damping import dispersion_atm

from ..cells import cells, param_on
from ..conftest import DEVICE
from ..utils import load_structure

# a cutoff below the default, which is what makes the sparse term pay off
cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=12.0)

tol = 1e-10

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _molecule(source: tuple[str, str]) -> Structure:
    return load_structure(*source, DD64)


def _param() -> dict[str, torch.Tensor | float]:
    return {**param_on(DD64), "s9": torch.tensor(1.0, **DD64)}


@pytest.mark.parametrize(
    "source", [("heavy28", "pbh4_bih3"), ("other", "C6H5I-CH3SH")]
)
def test_matches_dense(source: tuple[str, str]) -> None:
    structure = _molecule(source)
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_batch_matches_dense() -> None:
    structure = pack_structures(
        [
            _molecule(("heavy28", "pbh4_bih3")),
            _molecule(("other", "C6H5I-CH3SH")),
        ]
    )
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_fully_sparse_matches_dense() -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_wide_list_matches_dense() -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3 + 4.0, skin=1.0)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("max_triples", [50, 2_000_000])
def test_chunked_gradient_matches_dense(
    checkpoint: bool, max_triples: int
) -> None:
    structure = _molecule(("other", "C6H5I-CH3SH"))
    nat = structure.numbers.shape[-1]
    nbl = build_neighborlist(structure, cutoff.disp3)
    gen = torch.Generator().manual_seed(0)
    c6 = torch.rand(nat, nat, generator=gen, dtype=torch.double) + 1.0
    c6 = (0.5 * (c6 + c6.T)).to(DEVICE)  # C6 of a pair is symmetric

    def grad(**kwargs: object) -> torch.Tensor:
        pos = structure.positions.clone().requires_grad_(True)
        energy = dispersion_atm(
            structure.replace(positions=pos),
            c6,
            cutoff=cutoff.disp3,
            **kwargs,
        ).sum()
        return torch.autograd.grad(energy, pos)[0]

    sparse = grad(nbl=nbl, max_triples=max_triples, checkpoint=checkpoint)
    assert pytest.approx(grad().cpu(), abs=1e-9) == sparse.cpu()


def test_cell_raises() -> None:
    structure = cells["urea"]
    nbl = build_neighborlist(structure, cutoff.disp3)
    with pytest.raises(ValueError):
        dispersion_atm(structure, torch.ones(8, 8, **DD64), nbl=nbl)
