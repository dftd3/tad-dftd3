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
against the dense evaluation, for molecules and for periodic cells, where
the neighbours of a triple are images at lattice shifts. The cells are also
checked against s-dftd3 through the dense term (``test/test_periodic/``).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.neighbor.list import (
    NeighborList,
    build_neighborlist,
    build_neighborlists,
)
from tad_mctc.typing import DD

from tad_dftd3 import DampingParam, dftd3
from tad_dftd3 import disp as disp_module
from tad_dftd3 import sparse as sparse_module
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.damping import dispersion_atm, dispersion_atm_periodic
from tad_dftd3.model import AtomicC6
from tad_dftd3.sparse import dispersion_atm_sparse

from ..cells import TRICLINIC, cells, param_on, random_cell
from ..conftest import DEVICE
from ..reference import reference_energy_per_atom
from ..utils import load_structure

# a cutoff below the default, which is what makes the sparse term pay off
cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=12.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}

ATM = DampingParam(s9=1.0)
"""The three-body term with its default damping parameters."""


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
    c6 = torch.rand(
        nat, nat, generator=gen, dtype=torch.double, device="cpu"
    )
    c6 = c6 + 1.0
    c6 = (0.5 * (c6 + c6.T)).to(DEVICE)  # C6 of a pair is symmetric

    def grad(sparse: bool) -> torch.Tensor:
        pos = structure.positions.clone().requires_grad_(True)
        moved = structure.replace(positions=pos)
        if sparse:
            energy = dispersion_atm_sparse(
                moved,
                c6,
                ATM,
                nbl,
                cutoff=cutoff.disp3,
                max_triples=max_triples,
                checkpoint=checkpoint,
            )
        else:
            energy = dispersion_atm(moved, c6, ATM, cutoff=cutoff.disp3)
        return torch.autograd.grad(energy.sum(), pos)[0]

    assert pytest.approx(grad(False).cpu(), abs=1e-12) == grad(True).cpu()


@pytest.mark.parametrize(
    "name", ["urea", "nacl", "periodic_triclinic", "periodic_one_atom"]
)
def test_cell_matches_dense(name: str) -> None:
    structure = cells[name].to(DEVICE)
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == sparse.cpu()

    # the three-body term itself, and not only through the energy
    assert not torch.allclose(
        sparse, dftd3(structure, {**p, "s9": 0.0}, cutoff=cutoff), atol=1e-8
    )


@pytest.mark.parametrize("skin", [0.0, 1.5])
def test_cell_matches_reference(skin: float) -> None:
    structure = cells["periodic_triclinic"].to(DEVICE)
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3 + skin, skin=skin)

    ref = reference_energy_per_atom(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == sparse.cpu()


def test_cell_mixed_with_shifts() -> None:
    """Dense shifts for the pair terms, a list for the three-body term."""
    structure = cells["urea"].to(DEVICE)
    assert structure.lattice is not None and structure.periodic is not None
    p = _param()
    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, max(cutoff.cn, cutoff.disp2)
    )
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    mixed = dftd3(structure, p, cutoff=cutoff, shifts=shifts, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == mixed.cpu()


def test_cell_batch_matches_dense() -> None:
    """Different lattices, so the shifts are translated per system."""
    structure = pack_structures(
        [
            cells["urea"].to(DEVICE),
            cells["periodic_triclinic"].to(DEVICE),
            random_cell(TRICLINIC, 3, DD64, seed=4),
        ]
    )
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == sparse.cpu()


def test_cell_batch_shared_lattice() -> None:
    first = random_cell(TRICLINIC, 3, DD64, seed=8)
    second = first.replace(
        positions=random_cell(TRICLINIC, 3, DD64, seed=9).positions
    )
    batch = Structure(
        numbers=torch.stack([first.numbers, second.numbers]),
        positions=torch.stack([first.positions, second.positions]),
        lattice=first.lattice,
    )
    p = _param()
    nbl = build_neighborlist(batch, cutoff.disp3)

    dense = dftd3(batch, p, cutoff=cutoff)
    sparse = dftd3(batch, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == sparse.cpu()


@pytest.mark.parametrize(
    "periodic", [[True, True, False], [True, False, False]]
)
def test_low_dimensional_matches_dense(periodic: list[bool]) -> None:
    mask = torch.tensor(periodic, device=DEVICE)
    structure = random_cell(TRICLINIC, 4, DD64, seed=5, periodic=mask)
    p = _param()
    nbl = build_neighborlist(structure, cutoff.disp3)

    dense = dftd3(structure, p, cutoff=cutoff)
    sparse = dftd3(structure, p, cutoff=cutoff, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == sparse.cpu()


def test_cell_smooth_cutoff_matches_dense() -> None:
    cut = Cutoff(cn=15.0, disp2=20.0, disp3=12.0, width3=4.0)
    structure = cells["periodic_triclinic"].to(DEVICE)
    p = _param()
    nbl = build_neighborlist(structure, cut.disp3)

    dense = dftd3(structure, p, cutoff=cut)
    sparse = dftd3(structure, p, cutoff=cut, nbl_disp3=nbl)
    assert pytest.approx(dense.cpu(), abs=tol, rel=0) == sparse.cpu()


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("max_triples", [50, 2_000_000])
def test_cell_gradient_and_strain_match_dense(
    checkpoint: bool, max_triples: int
) -> None:
    """Gradient with respect to the positions and to the lattice."""
    structure = cells["periodic_triclinic"].to(DEVICE)
    assert structure.lattice is not None
    nat = structure.numbers.shape[-1]
    nbl = build_neighborlist(structure, cutoff.disp3)
    gen = torch.Generator().manual_seed(0)
    c6 = torch.rand(
        nat, nat, generator=gen, dtype=torch.double, device="cpu"
    )
    c6 = c6 + 1.0
    c6 = (0.5 * (c6 + c6.T)).to(DEVICE)

    def grads(sparse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        pos = structure.positions.clone().requires_grad_(True)
        lat = structure.lattice.clone().requires_grad_(True)  # type: ignore[union-attr]
        moved = structure.replace(positions=pos, lattice=lat)
        if sparse:
            energy = dispersion_atm_sparse(
                moved,
                c6,
                ATM,
                nbl,
                cutoff=cutoff.disp3,
                max_triples=max_triples,
                checkpoint=checkpoint,
            )
        else:
            energy = dispersion_atm_periodic(
                moved, c6, ATM, cutoff=cutoff.disp3
            )
        grad_pos, grad_lat = torch.autograd.grad(energy.sum(), (pos, lat))
        return grad_pos, grad_lat

    dense = grads(False)
    sparse = grads(True)
    for d, s in zip(dense, sparse):
        assert pytest.approx(d.cpu(), abs=1e-12, rel=0) == s.cpu()


def test_list_takes_the_place_of_shifts() -> None:
    """
    `shifts` serve the terms without a list: with lists for the pairs, only
    the dense three-body term, and with `sparse`, which builds a list for
    the three-body term too, none.
    """
    structure = cells["urea"].to(DEVICE)
    assert structure.lattice is not None and structure.periodic is not None
    p = _param()
    ref = dftd3(structure, p, cutoff=cutoff)

    # too short for the coordination number and the two-body term
    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff.disp3
    )
    with pytest.raises(ValueError, match="cutoff"):
        dftd3(structure, p, cutoff=cutoff, shifts=shifts)

    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2)
    )
    energy = dftd3(
        structure,
        p,
        cutoff=cutoff,
        shifts=shifts,
        nbl_cn=nbl_cn,
        nbl_disp2=nbl_disp2,
    )
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()

    # too short for every term
    short = build_periodic_shifts(structure.lattice, structure.periodic, 5.0)
    energy = dftd3(structure, p, cutoff=cutoff, shifts=short, sparse=True)
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


def test_checkpoint_gradient_matches_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `checkpoint` and `max_triples` reach every sparse term from `dftd3`:
    each chunk of pairs of the coordination number and the two-body term,
    and each chunk of triples, is recomputed, and the gradient is the same.
    """
    chunk = 64
    monkeypatch.setattr("tad_mctc.ncoord.common._CHUNK_SIZE_CPU", chunk)
    monkeypatch.setattr("tad_mctc.ncoord.common._CHUNK_SIZE_GPU", chunk)

    # the pair walk looks `torch.utils.checkpoint.checkpoint` up on each
    # call, the three-body term holds its own reference
    calls = {"pairs": 0, "triples": 0}

    def counting(key: str, function: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls[key] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        torch.utils.checkpoint,
        "checkpoint",
        counting("pairs", torch.utils.checkpoint.checkpoint),
    )
    monkeypatch.setattr(
        sparse_module,
        "_torch_checkpoint",
        counting("triples", sparse_module._torch_checkpoint),
    )

    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = _param()
    nbl_cn, nbl_disp2, nbl_disp3 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2, cutoff.disp3)
    )

    def grad(sparse: bool) -> torch.Tensor:
        pos = structure.positions.clone().requires_grad_(True)
        moved = structure.replace(positions=pos)
        if sparse:
            energy = dftd3(
                moved,
                p,
                cutoff=cutoff,
                nbl_cn=nbl_cn,
                nbl_disp2=nbl_disp2,
                nbl_disp3=nbl_disp3,
                max_triples=50,
                checkpoint=True,
            )
        else:
            energy = dftd3(moved, p, cutoff=cutoff)
        return torch.autograd.grad(energy.sum(), pos)[0]

    assert pytest.approx(grad(False).cpu(), abs=1e-12) == grad(True).cpu()

    def n_chunks(nbl: NeighborList) -> int:
        return -(-max(nbl.idx_i.shape[0], 1) // chunk)

    assert calls["pairs"] == n_chunks(nbl_cn) + n_chunks(nbl_disp2)
    assert calls["triples"] > 1


def test_no_triples_keeps_graph() -> None:
    """Without any triple, the energy is still a (zero) function of the
    positions, as the dense term."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], **DD64)
    pos.requires_grad_(True)
    structure = Structure(
        numbers=torch.tensor([1, 1], device=DEVICE), positions=pos
    )
    c6 = torch.ones(2, 2, **DD64)
    nbl = build_neighborlist(structure, cutoff.disp3)

    energy = dispersion_atm_sparse(structure, c6, ATM, nbl, cutoff=cutoff.disp3)
    (grad,) = torch.autograd.grad(energy.sum(), pos)
    assert (energy == 0).all() and (grad == 0).all()


@pytest.mark.parametrize("periodic", [False, True])
def test_sparse_builds_three_body_list(
    monkeypatch: pytest.MonkeyPatch, periodic: bool
) -> None:
    """With `sparse`, no term is dense, the three-body term included."""
    if periodic:
        structure = cells["urea"].to(DEVICE)
    else:
        structure = _molecule(("other", "C6H5I-CH3SH"))
    p = _param()
    dense = dftd3(structure, p, cutoff=cutoff)

    def fail(*_: Any, **__: Any) -> Any:
        raise AssertionError("dense three-body term")

    monkeypatch.setattr(disp_module, "dispersion_atm", fail)
    monkeypatch.setattr(disp_module, "dispersion_atm_periodic", fail)

    sparse = dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert pytest.approx(dense.cpu(), abs=tol) == sparse.cpu()


def test_c6_matrix_only_for_dense_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `dftd3` builds the C6 matrix once for the dense terms together, and not
    at all over lists: there, a lookup in it would create a gradient of its
    full size per chunk.
    """
    structure = _molecule(("other", "C6H5I-CH3SH"))
    p = _param()

    calls: list[int] = []
    dense = AtomicC6.dense

    def counting(self: AtomicC6) -> torch.Tensor:
        calls.append(1)
        return dense(self)

    monkeypatch.setattr(AtomicC6, "dense", counting)

    dftd3(structure, p, cutoff=cutoff)
    assert len(calls) == 1

    calls.clear()
    dftd3(structure, p, cutoff=cutoff, sparse=True)
    assert len(calls) == 0
