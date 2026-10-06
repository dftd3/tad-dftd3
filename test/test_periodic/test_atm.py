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
Dense periodic three-body (ATM) term against s-dftd3.

An atom of the cell forms a triple with every pair of atoms or images of
atoms within the cutoff of it, also with images of itself. The atom-resolved
energies are compared to s-dftd3's, reconstructed from its pairwise energies
(see ``test/reference.py``), and the derivatives to its gradient and virial.
The cutoff of the three-body term is kept short, as the number of triples
grows with its sixth power.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import DampingParam, dftd3, disp
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.damping import dispersion_atm, dispersion_atm_periodic

from ..cells import TRICLINIC, cells, param_on, random_cell
from ..conftest import DEVICE
from ..reference import reference_energy_per_atom, reference_gradient

cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=10.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}

ATM = DampingParam(s9=1.0)
"""The three-body term with its default damping parameters."""


def _param() -> dict[str, Tensor | float]:
    return {**param_on(DD64), "s9": torch.tensor(1.0, **DD64)}


def _check(structure: Structure, cut: Cutoff = cutoff) -> Tensor:
    """
    The energy of `structure`, after checking it against s-dftd3 at the
    same cutoffs, and that the three-body term is part of it.
    """
    p = _param()
    ref = reference_energy_per_atom(structure, p, cutoff=cut)
    energy = dftd3(structure, p, cutoff=cut)

    assert energy.shape == structure.numbers.shape
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()

    two_body = dftd3(structure, {**p, "s9": 0.0}, cutoff=cut)
    assert not torch.allclose(energy, two_body, atol=1e-8, rtol=0)
    return energy


@pytest.mark.parametrize("name", list(cells))
def test_crystal(name: str) -> None:
    _check(cells[name].to(DEVICE))


def test_one_atom_cell() -> None:
    """
    Every triple is of an atom with its own images, which is where the
    weight of a triple of equal atoms (a sixth) decides the result.
    """
    structure = cells["periodic_one_atom"].to(DEVICE)
    assert structure.numbers.shape[-1] == 1
    _check(structure)


def test_atom_with_its_images() -> None:
    """A cell smaller than the cutoff, in which atoms see their images."""
    lattice = torch.diag(torch.tensor([4.0, 4.5, 5.0], dtype=torch.double))
    _check(random_cell(lattice, 3, DD64, seed=2))


@pytest.mark.parametrize("width3", [2.0, 5.0])
def test_smooth_cutoff(width3: float) -> None:
    cut = Cutoff(cn=15.0, disp2=20.0, disp3=10.0, width3=width3)
    structure = cells["periodic_triclinic"].to(DEVICE)

    energy = _check(structure, cut)
    assert not torch.allclose(
        energy, dftd3(structure, _param(), cutoff=cutoff), atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic", "nacl"])
def test_unwrapped_positions(name: str) -> None:
    """Atoms moved by whole lattice vectors: the same crystal."""
    structure = cells[name].to(DEVICE)
    assert structure.lattice is not None

    generator = torch.Generator().manual_seed(0)
    offsets = torch.randint(
        -5, 6, structure.positions.shape, generator=generator, device="cpu"
    )
    unwrapped = structure.positions + offsets.to(**DD64) @ structure.lattice

    energy = _check(structure.replace(positions=unwrapped))
    wrapped = dftd3(structure, _param(), cutoff=cutoff)
    assert pytest.approx(wrapped.cpu(), abs=tol, rel=0) == energy.cpu()


@pytest.mark.parametrize(
    "periodic",
    [
        [True, True, False],
        [False, True, True],
        [True, False, False],
        [False, False, True],
    ],
)
def test_low_dimensional(periodic: list[bool]) -> None:
    """Slabs and chains, with the atoms of the open axes inside the cell."""
    mask = torch.tensor(periodic, device=DEVICE)
    _check(random_cell(TRICLINIC, 5, DD64, seed=3, periodic=mask))


def test_no_periodic_axis_is_a_molecule() -> None:
    mask = torch.tensor([False, False, False], device=DEVICE)
    structure = random_cell(TRICLINIC, 5, DD64, seed=6, periodic=mask)
    molecule = Structure(
        numbers=structure.numbers, positions=structure.positions
    )

    energy = _check(structure)
    assert pytest.approx(_check(molecule).cpu(), abs=tol, rel=0) == energy.cpu()


def test_batch() -> None:
    """Different cells, in a padded batch: padding is exactly zero."""
    systems = [
        cells["periodic_triclinic"].to(DEVICE),
        random_cell(TRICLINIC, 3, DD64, seed=7),
        random_cell(
            TRICLINIC,
            4,
            DD64,
            seed=8,
            periodic=torch.tensor([True, True, False], device=DEVICE),
        ),
    ]
    batch = pack_structures(systems)

    energy = dftd3(batch, _param(), cutoff=cutoff)

    assert energy.shape == batch.numbers.shape
    for i, system in enumerate(systems):
        nat = system.numbers.shape[-1]
        ref = reference_energy_per_atom(system, _param(), cutoff=cutoff)
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i, :nat].cpu()
        assert (energy[i, nat:] == 0).all()


def test_batch_shared_lattice() -> None:
    first = random_cell(TRICLINIC, 3, DD64, seed=8)
    second = first.replace(
        positions=random_cell(TRICLINIC, 3, DD64, seed=9).positions
    )
    batch = Structure(
        numbers=torch.stack([first.numbers, second.numbers]),
        positions=torch.stack([first.positions, second.positions]),
        lattice=first.lattice,
    )

    energy = dftd3(batch, _param(), cutoff=cutoff)
    for i, system in enumerate([first, second]):
        ref = reference_energy_per_atom(system, _param(), cutoff=cutoff)
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i].cpu()


@pytest.mark.parametrize("max_triples", [1, 50, 10_000])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_chunks(max_triples: int, checkpoint: bool) -> None:
    """The size of a block of triples and checkpointing change nothing."""
    structure = cells["periodic_triclinic"].to(DEVICE)
    c6 = disp.D3Model(cutoff=cutoff).c6(structure)

    ref = dispersion_atm_periodic(structure, c6, ATM, cutoff=cutoff.disp3)
    energy = dispersion_atm_periodic(
        structure,
        c6,
        ATM,
        cutoff=cutoff.disp3,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )
    assert pytest.approx(ref.cpu(), abs=1e-15, rel=0) == energy.cpu()


def test_shifts_given() -> None:
    """A table covering more than the cutoff gives the same energy."""
    structure = cells["periodic_triclinic"].to(DEVICE)
    assert structure.lattice is not None and structure.periodic is not None
    shifts = build_periodic_shifts(structure.lattice, structure.periodic, 20.0)

    ref = dftd3(structure, _param(), cutoff=cutoff)
    energy = dftd3(structure, _param(), cutoff=cutoff, shifts=shifts)
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


def test_shifts_too_short() -> None:
    structure = cells["periodic_triclinic"].to(DEVICE)
    assert structure.lattice is not None and structure.periodic is not None
    shifts = build_periodic_shifts(structure.lattice, structure.periodic, 5.0)
    c6 = disp.D3Model(cutoff=cutoff).c6(structure)

    with pytest.raises(ValueError):
        dispersion_atm_periodic(structure, c6, ATM, shifts, cutoff=cutoff.disp3)


def test_shifts_without_lattice() -> None:
    cell = cells["periodic_triclinic"].to(DEVICE)
    assert cell.lattice is not None and cell.periodic is not None
    molecule = Structure(numbers=cell.numbers, positions=cell.positions)
    shifts = build_periodic_shifts(cell.lattice, cell.periodic, 10.0)
    c6 = torch.ones(*cell.numbers.shape, cell.numbers.shape[-1], **DD64)

    with pytest.raises(ValueError, match="lattice"):
        dispersion_atm_periodic(molecule, c6, ATM, shifts)

    # also through the dispatch, where shifts select the periodic path
    with pytest.raises(ValueError, match="lattice"):
        disp.dispersion3(molecule, _param(), c6, pairs=shifts)


def test_wrong_function_for_the_system() -> None:
    """The dense functions do not guess: a cell is not a molecule."""
    cell = cells["periodic_triclinic"].to(DEVICE)
    molecule = Structure(numbers=cell.numbers, positions=cell.positions)
    c6 = torch.ones(*cell.numbers.shape, cell.numbers.shape[-1], **DD64)

    with pytest.raises(ValueError, match="dispersion_atm_periodic"):
        dispersion_atm(cell, c6, ATM)
    with pytest.raises(ValueError, match="lattice"):
        dispersion_atm_periodic(molecule, c6, ATM)


def test_s9_zero_skips_three_body() -> None:
    structure = cells["urea"].to(DEVICE)
    p = _param()

    off = dftd3(structure, {**p, "s9": 0.0}, cutoff=cutoff)
    ref = reference_energy_per_atom(structure, {**p, "s9": 0.0}, cutoff=cutoff)
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == off.cpu()


@pytest.mark.parametrize("name", ["urea", "periodic_one_atom"])
def test_gradient_and_virial(name: str) -> None:
    """Position gradient and virial (strain derivative) against s-dftd3."""
    structure = cells[name].to(DEVICE)
    assert structure.lattice is not None
    p = _param()

    strain = torch.zeros(3, 3, requires_grad=True, **DD64)
    positions = structure.positions.clone().requires_grad_(True)
    strained = structure.replace(
        positions=positions + positions @ strain,
        lattice=structure.lattice + structure.lattice @ strain,
    )
    energy = dftd3(strained, p, cutoff=cutoff).sum()
    grad, virial = torch.autograd.grad(energy, (positions, strain))

    ref_grad, ref_virial = reference_gradient(structure, p, cutoff=cutoff)
    assert pytest.approx(ref_grad.cpu(), abs=1e-10, rel=0) == grad.cpu()
    assert pytest.approx(ref_virial.cpu(), abs=1e-10, rel=0) == virial.cpu()


def test_no_triples_keeps_graph() -> None:
    """Without any triple, the energy is still a (zero) function of the
    positions, as for a molecule."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], **DD64)
    pos.requires_grad_(True)
    structure = Structure(
        numbers=torch.tensor([1, 1], device=DEVICE),
        positions=pos,
        lattice=30.0 * torch.eye(3, **DD64),
        periodic=torch.tensor([True, True, True], device=DEVICE),
    )
    c6 = torch.ones(2, 2, **DD64)

    # shorter than every distance
    energy = dispersion_atm_periodic(structure, c6, ATM, cutoff=1.0)
    (grad,) = torch.autograd.grad(energy.sum(), pos)
    assert (energy == 0).all() and (grad == 0).all()
