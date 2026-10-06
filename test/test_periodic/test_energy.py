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
Dense periodic dispersion energy against s-dftd3.

Every atom is paired with every periodic image of every atom, for the
coordination number and the two-body term. The atom-resolved energies are
compared to s-dftd3's, reconstructed from its pairwise energies (see
``test/reference.py``).

With ``cutoff=None`` both implementations use their own default cutoffs, so
those tests also check that the defaults agree for a cell, where the
cutoffs decide which images contribute.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import dftd3, disp, model, ncoord
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.reference import Reference

from ..cells import TRICLINIC, cells, param, param_on, random_cell
from ..conftest import DEVICE
from ..reference import reference_energy_per_atom, reference_pairwise

# short enough to cut images of the cells below, so that it matters
cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _check(structure: Structure, cut: Cutoff | None = cutoff) -> Tensor:
    """
    The energy of `structure`, after checking it against s-dftd3 at the
    same cutoffs (``None`` for both at their own defaults).
    """
    ref = reference_energy_per_atom(structure, param, cutoff=cut)
    energy = dftd3(structure, param_on(DD64), cutoff=cut or Cutoff())

    assert energy.shape == structure.numbers.shape
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()
    return energy


@pytest.mark.parametrize("name", list(cells))
def test_crystal_default_cutoffs(name: str) -> None:
    """Both implementations at their own (default) cutoffs."""
    _check(cells[name].to(DEVICE), cut=None)


@pytest.mark.parametrize("name", list(cells))
def test_crystal_changed_cutoff(name: str) -> None:
    """Both implementations at the same, non-default cutoffs."""
    structure = cells[name].to(DEVICE)
    energy = _check(structure)

    # only meaningful if the cutoff changes the result
    default = dftd3(structure, param_on(DD64))
    assert not torch.allclose(energy, default, atol=1e-8, rtol=0)


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_crystal_float32(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float}
    structure = cells[name].to(DEVICE)

    ref = reference_energy_per_atom(structure, param, cutoff=cutoff)
    energy = dftd3(structure.to(**dd), param_on(dd), cutoff=cutoff)

    assert energy.dtype == torch.float
    assert pytest.approx(ref.cpu(), abs=1e-6, rel=1e-5) == energy.cpu()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_self_images(seed: int) -> None:
    """A cell so small that every atom interacts with its own images."""
    lattice = 4.5 * torch.eye(3, dtype=torch.double)
    _check(random_cell(lattice, 3, DD64, seed=seed))


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_two_body_pairwise(name: str) -> None:
    """
    The two-body term alone, from a given C6, against s-dftd3's additive
    pairwise energies.
    """
    structure = cells[name].to(DEVICE)

    # the C6 of the full model, so that s-dftd3 sees the same
    cn = ncoord.cn_d3.replace(cutoff=cutoff.cn)(structure)
    ref = Reference.load(**DD64)
    weights = model.weight_references(structure.numbers, cn, ref)
    c6 = model.atomic_c6(structure.numbers, weights, ref)

    two_body, _ = reference_pairwise(structure, param, cutoff=cutoff)
    energy = disp.dispersion2(structure, param_on(DD64), c6, cutoff=cutoff)

    assert pytest.approx(two_body.sum(-1).cpu(), abs=tol, rel=0) == energy.cpu()


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic", "nacl"])
def test_unwrapped_positions(name: str) -> None:
    """
    Atoms moved by whole lattice vectors, far outside the cell: the same
    crystal, so the same energy, also in s-dftd3.
    """
    structure = cells[name].to(DEVICE)
    assert structure.lattice is not None

    generator = torch.Generator().manual_seed(0)
    offsets = torch.randint(
        -5, 6, structure.positions.shape, generator=generator, device="cpu"
    )
    unwrapped = structure.positions + offsets.to(**DD64) @ structure.lattice

    energy = _check(structure.replace(positions=unwrapped))
    wrapped = dftd3(structure, param_on(DD64), cutoff=cutoff)
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
@pytest.mark.parametrize("seed", [3, 4])
def test_low_dimensional(periodic: list[bool], seed: int) -> None:
    """
    Slabs and chains of a triclinic cell. Along the periodic axes the atoms
    lie outside the cell; along the open axes inside it, because s-dftd3
    folds those too (see ``test/reference.py``).
    """
    mask = torch.tensor(periodic, device=DEVICE)
    structure = random_cell(TRICLINIC, 6, DD64, seed=seed, periodic=mask)
    assert structure.lattice is not None

    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randint(
        -3, 4, structure.positions.shape, generator=generator, device="cpu"
    )
    offsets = torch.where(mask.cpu(), offsets, torch.zeros_like(offsets))
    unwrapped = structure.positions + offsets.to(**DD64) @ structure.lattice

    _check(structure.replace(positions=unwrapped))


def test_short_vector_along_open_axis() -> None:
    """
    A slab whose open axis has a short placeholder lattice vector: its
    images would be within the cutoff, but do not exist.
    """
    lattice = torch.diag(torch.tensor([7.0, 7.5, 2.0], dtype=torch.double))
    mask = torch.tensor([True, True, False], device=DEVICE)
    _check(random_cell(lattice, 5, DD64, seed=5, periodic=mask))


def test_no_periodic_axis_is_a_molecule() -> None:
    mask = torch.tensor([False, False, False], device=DEVICE)
    structure = random_cell(TRICLINIC, 6, DD64, seed=6, periodic=mask)
    molecule = Structure(
        numbers=structure.numbers, positions=structure.positions
    )

    energy = _check(structure)
    assert pytest.approx(_check(molecule).cpu(), abs=tol, rel=0) == energy.cpu()


def test_batch() -> None:
    """
    Cells of different shapes, sizes and periodicity in one padded batch:
    each is its own energy, and the padding is exactly zero.
    """
    systems = [
        cells["urea"].to(DEVICE),
        cells["periodic_triclinic"].to(DEVICE),
        random_cell(
            TRICLINIC,
            4,
            DD64,
            seed=7,
            periodic=torch.tensor([True, True, False], device=DEVICE),
        ),
    ]
    batch = pack_structures(systems)

    energy = dftd3(batch, param_on(DD64), cutoff=cutoff)

    assert energy.shape == batch.numbers.shape
    for i, system in enumerate(systems):
        nat = system.numbers.shape[-1]
        ref = reference_energy_per_atom(system, param, cutoff=cutoff)
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i, :nat].cpu()
        assert (energy[i, nat:] == 0).all()


def test_batch_shared_lattice() -> None:
    """A batch of geometries in the same cell, with one ``(3, 3)`` lattice."""
    first = random_cell(TRICLINIC, 4, DD64, seed=8)
    second = first.replace(
        positions=random_cell(TRICLINIC, 4, DD64, seed=9).positions
    )
    batch = Structure(
        numbers=torch.stack([first.numbers, second.numbers]),
        positions=torch.stack([first.positions, second.positions]),
        lattice=first.lattice,
    )

    energy = dftd3(batch, param_on(DD64), cutoff=cutoff)

    for i, system in enumerate([first, second]):
        ref = reference_energy_per_atom(system, param, cutoff=cutoff)
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i].cpu()


def test_given_shifts() -> None:
    """
    Shifts built beforehand give the same energy, also if they cover more
    images than needed.
    """
    structure = cells["urea"].to(DEVICE)
    assert structure.lattice is not None and structure.periodic is not None

    ref = dftd3(structure, param_on(DD64), cutoff=cutoff)

    for table_cutoff in (cutoff.disp2, 2 * cutoff.disp2):
        shifts = build_periodic_shifts(
            structure.lattice, structure.periodic, table_cutoff
        )
        energy = dftd3(structure, param_on(DD64), shifts=shifts, cutoff=cutoff)
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


def test_three_body_float_zero() -> None:
    """`s9` as the Python number 0.0 skips the three-body term."""
    structure = cells["diamond"].to(DEVICE)

    ref = dftd3(structure, param_on(DD64), cutoff=cutoff)

    for s9 in (0.0, torch.tensor(0.0, **DD64)):
        # A zero tensor outside of any transform is skipped as well.
        energy = dftd3(structure, {**param_on(DD64), "s9": s9}, cutoff=cutoff)
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()
