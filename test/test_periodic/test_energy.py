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
from tad_mctc.batch import pack
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import dftd3, disp, model, ncoord
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.reference import Reference

from ..cells import cells, random_cell
from ..conftest import DEVICE
from ..reference import reference_energy_per_atom, reference_pairwise

# TPSS0-D3(BJ), without the three-body term, which is not periodic
param = {
    "s6": torch.tensor(1.0000, dtype=torch.double),
    "s8": torch.tensor(1.2576, dtype=torch.double),
    "a1": torch.tensor(0.3768, dtype=torch.double),
    "a2": torch.tensor(4.5865, dtype=torch.double),
}

# short enough to cut images of the cells below, so that it matters
cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-12

ALL_AXES = [True, True, True]

TRICLINIC = torch.tensor(
    [[7.0, 0.0, 0.0], [1.2, 6.5, 0.0], [0.6, 0.9, 6.0]], dtype=torch.double
)


def _param(dd: DD) -> dict[str, Tensor | float]:
    return {k: v.to(**dd) for k, v in param.items()}


@pytest.mark.parametrize("name", list(cells))
def test_crystal_default_cutoffs(name: str) -> None:
    """Both implementations at their own (default) cutoffs."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    ref = reference_energy_per_atom(numbers, positions, param, lattice=lattice)
    energy = dftd3(numbers, positions, _param(dd), lattice=lattice)

    assert energy.shape == numbers.shape
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


@pytest.mark.parametrize("name", list(cells))
def test_crystal_changed_cutoff(name: str) -> None:
    """Both implementations at the same, non-default cutoffs."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    ref = reference_energy_per_atom(
        numbers, positions, param, cutoff=cutoff, lattice=lattice
    )
    energy = dftd3(
        numbers, positions, _param(dd), lattice=lattice, cutoff=cutoff
    )

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()

    # only meaningful if the cutoff changes the result
    default = dftd3(numbers, positions, _param(dd), lattice=lattice)
    assert not torch.allclose(energy, default, atol=1e-8, rtol=0)


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_crystal_float32(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    ref = reference_energy_per_atom(
        numbers, positions, param, cutoff=cutoff, lattice=lattice
    )
    energy = dftd3(
        numbers,
        positions.to(**dd),
        _param(dd),
        lattice=lattice.to(**dd),
        cutoff=cutoff,
    )

    assert energy.dtype == torch.float
    assert pytest.approx(ref.cpu(), abs=1e-6, rel=1e-5) == energy.cpu()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_self_images(seed: int) -> None:
    """A cell so small that every atom interacts with its own images."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = random_cell(
        4.5 * torch.eye(3, dtype=torch.double), 3, dd, seed=seed
    )

    ref = reference_energy_per_atom(
        numbers, positions, param, cutoff=cutoff, lattice=lattice
    )
    energy = dftd3(
        numbers, positions, _param(dd), lattice=lattice, cutoff=cutoff
    )

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_two_body_pairwise(name: str) -> None:
    """
    The two-body term alone, from a given C6, against s-dftd3's additive
    pairwise energies.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    # the C6 of the full model, so that s-dftd3 sees the same
    c6 = _atomic_c6(numbers, positions, lattice, cutoff)

    two_body, _ = reference_pairwise(
        numbers, positions, param, cutoff=cutoff, lattice=lattice
    )
    energy = disp.dispersion2(
        numbers,
        positions,
        _param(dd),
        c6,
        lattice=lattice,
        cutoff=cutoff.disp2,
    )

    assert pytest.approx(two_body.sum(-1).cpu(), abs=tol, rel=0) == energy.cpu()


def _atomic_c6(
    numbers: Tensor, positions: Tensor, lattice: Tensor, cutoff: Cutoff
) -> Tensor:
    """C6 of a cell from the full pipeline of :func:`dftd3`."""
    cn_model = ncoord.cn_d3.replace(cutoff=cutoff.cn)
    structure = Structure(numbers=numbers, positions=positions, lattice=lattice)
    cn = cn_model(structure)

    ref = Reference(device=positions.device, dtype=positions.dtype)
    weights = model.weight_references(numbers, cn, ref)
    return model.atomic_c6(numbers, weights, ref)


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic", "nacl"])
def test_unwrapped_positions(name: str) -> None:
    """
    Atoms moved by whole lattice vectors, far outside the cell: the same
    crystal, so the same energy, also in s-dftd3.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells[name])

    generator = torch.Generator().manual_seed(0)
    offsets = torch.randint(-5, 6, positions.shape, generator=generator)
    unwrapped = positions + offsets.to(**dd) @ lattice

    ref = reference_energy_per_atom(
        numbers, unwrapped, param, cutoff=cutoff, lattice=lattice
    )
    energy = dftd3(
        numbers, unwrapped, _param(dd), lattice=lattice, cutoff=cutoff
    )
    wrapped = dftd3(
        numbers, positions, _param(dd), lattice=lattice, cutoff=cutoff
    )

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()
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
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = random_cell(TRICLINIC, 6, dd, seed=seed)
    mask = torch.tensor(periodic, device=DEVICE)

    generator = torch.Generator().manual_seed(seed)
    offsets = torch.randint(-3, 4, positions.shape, generator=generator)
    offsets = torch.where(mask.cpu(), offsets, torch.zeros_like(offsets))
    positions = positions + offsets.to(**dd) @ lattice

    ref = reference_energy_per_atom(
        numbers,
        positions,
        param,
        cutoff=cutoff,
        lattice=lattice,
        periodic=mask,
    )
    energy = dftd3(
        numbers,
        positions,
        _param(dd),
        lattice=lattice,
        periodic=mask,
        cutoff=cutoff,
    )

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


def test_short_vector_along_open_axis() -> None:
    """
    A slab whose open axis has a short placeholder lattice vector: its
    images would be within the cutoff, but do not exist.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    lattice = torch.diag(torch.tensor([7.0, 7.5, 2.0], dtype=torch.double))
    numbers, positions, lattice = random_cell(lattice, 5, dd, seed=5)
    mask = torch.tensor([True, True, False], device=DEVICE)

    ref = reference_energy_per_atom(
        numbers,
        positions,
        param,
        cutoff=cutoff,
        lattice=lattice,
        periodic=mask,
    )
    energy = dftd3(
        numbers,
        positions,
        _param(dd),
        lattice=lattice,
        periodic=mask,
        cutoff=cutoff,
    )

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


def test_no_periodic_axis_is_a_molecule() -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = random_cell(TRICLINIC, 6, dd, seed=6)
    mask = torch.tensor([False, False, False], device=DEVICE)

    energy = dftd3(
        numbers,
        positions,
        _param(dd),
        lattice=lattice,
        periodic=mask,
        cutoff=cutoff,
    )
    molecule = dftd3(numbers, positions, _param(dd), cutoff=cutoff)
    ref = reference_energy_per_atom(numbers, positions, param, cutoff=cutoff)

    assert pytest.approx(molecule.cpu(), abs=tol, rel=0) == energy.cpu()
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


def test_batch() -> None:
    """
    Cells of different shapes, sizes and periodicity in one padded batch:
    each is its own energy, and the padding is exactly zero.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    systems = [
        (*(t.to(DEVICE) for t in cells["urea"]), ALL_AXES),
        (*(t.to(DEVICE) for t in cells["periodic_triclinic"]), ALL_AXES),
        (*random_cell(TRICLINIC, 4, dd, seed=7), [True, True, False]),
    ]

    numbers = pack([s[0] for s in systems])
    positions = pack([s[1] for s in systems])
    lattice = torch.stack([s[2] for s in systems])
    periodic = torch.tensor([s[3] for s in systems], device=DEVICE)

    energy = dftd3(
        numbers,
        positions,
        _param(dd),
        lattice=lattice,
        periodic=periodic,
        cutoff=cutoff,
    )

    assert energy.shape == numbers.shape
    for i, (n, p, lat, mask) in enumerate(systems):
        nat = n.shape[-1]
        ref = reference_energy_per_atom(
            n,
            p,
            param,
            cutoff=cutoff,
            lattice=lat,
            periodic=torch.tensor(mask, device=DEVICE),
        )
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i, :nat].cpu()
        assert (energy[i, nat:] == 0).all()


def test_batch_shared_lattice() -> None:
    """A batch of geometries in the same cell, with one ``(3, 3)`` lattice."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers0, positions0, lattice = random_cell(TRICLINIC, 4, dd, seed=8)
    _, positions1, _ = random_cell(TRICLINIC, 4, dd, seed=9)

    numbers = torch.stack([numbers0, numbers0])
    positions = torch.stack([positions0, positions1])

    energy = dftd3(
        numbers, positions, _param(dd), lattice=lattice, cutoff=cutoff
    )

    for i in range(2):
        ref = reference_energy_per_atom(
            numbers[i], positions[i], param, cutoff=cutoff, lattice=lattice
        )
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy[i].cpu()


def test_given_shifts() -> None:
    """
    Shifts built beforehand give the same energy, also if they cover more
    images than needed.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells["urea"])
    periodic = torch.tensor(ALL_AXES, device=DEVICE)

    ref = dftd3(numbers, positions, _param(dd), lattice=lattice, cutoff=cutoff)

    for table_cutoff in (cutoff.disp2, 2 * cutoff.disp2):
        shifts = build_periodic_shifts(lattice, periodic, table_cutoff)
        energy = dftd3(
            numbers,
            positions,
            _param(dd),
            lattice=lattice,
            shifts=shifts,
            cutoff=cutoff,
        )
        assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()


def test_three_body_float_zero() -> None:
    """`s9` as the Python number 0.0 skips the three-body term."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    numbers, positions, lattice = (t.to(DEVICE) for t in cells["diamond"])

    ref = dftd3(numbers, positions, _param(dd), lattice=lattice, cutoff=cutoff)
    energy = dftd3(
        numbers,
        positions,
        {**_param(dd), "s9": 0.0},
        lattice=lattice,
        cutoff=cutoff,
    )

    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()

    # a zero tensor outside of any transform is skipped as well
    energy = dftd3(
        numbers,
        positions,
        {**_param(dd), "s9": torch.tensor(0.0, **dd)},
        lattice=lattice,
        cutoff=cutoff,
    )
    assert pytest.approx(ref.cpu(), abs=tol, rel=0) == energy.cpu()
