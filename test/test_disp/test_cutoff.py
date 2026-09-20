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
Test that the real-space cutoffs match the s-dftd3 Fortran reference.

Every other module here compares against s-dftd3 on molecules that fit
inside every cutoff, where a wrong cutoff changes nothing. The geometries
below are larger than the cutoffs, so each of the three -- coordination
number, two-body, three-body -- decides which pairs and triples contribute.

Two things make these tests bite:

1. The reference is asked for *its own* cutoffs, so a wrong default here is
   a disagreement, not a shared convention (see ``test/reference.py``).
2. Every comparison is paired with a sensitivity check, so none can pass
   because the cutoff was irrelevant.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tad_mctc.io.read import read
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import defaults, dftd3
from tad_dftd3.cutoff import Cutoff

from ..conftest import DEVICE
from ..reference import reference_energy_per_atom
from ..samples import samples as mols

# TPSS0-D3BJ-ATM parameters, so that both the two- and the three-body term
# contribute.
param = {
    "s6": torch.tensor(1.0000),
    "s8": torch.tensor(1.2576),
    "s9": torch.tensor(1.0000),
    "alp": torch.tensor(14.00),
    "a1": torch.tensor(0.3768),
    "a2": torch.tensor(4.5865),
}

FRAGMENT_COUNT = 5
"""Number of fragments :func:`fragment_chain` places in a row."""

FRAGMENT_SPACING = 14.0
"""
Distance between neighbouring fragments in :func:`fragment_chain`, in Bohr.

Five fragments at this spacing sit 14, 28, 42 and 56 a0 apart, one pair
distance per window that tells the cutoffs apart: 28 a0 is inside the
40 a0 coordination-number cutoff but outside a 26 a0 one, 42 a0 inside a
50 a0 three-body cutoff but outside the default 40 a0 one, and 56 a0 inside
the 60 a0 two-body cutoff but outside a 50 a0 one.
"""

CHANGED_CUTOFFS = [("cn", 26.0), ("disp2", 50.0), ("disp3", 50.0)]
"""
One changed value per cutoff, each visible to this geometry.

Two narrow the default and ``disp3`` widens it (40 -> 50 a0 *adds* the
triples with a 42 a0 side); only that the energy changes matters, not the
direction. Each value sits in a gap between pair distances, so no
comparison depends on how a distance equal to the cutoff is rounded.
"""


def fragment_chain(name: str, dd: DD) -> tuple[Tensor, Tensor]:
    """
    Build :data:`FRAGMENT_COUNT` copies of molecule ``name`` in a row along
    x, :data:`FRAGMENT_SPACING` Bohr apart.

    Parameters
    ----------
    name : str
        Name of the molecule in :data:`test.samples.samples`.
    dd : DD
        Device and dtype of the returned positions.

    Returns
    -------
    (Tensor, Tensor)
        Atomic numbers, shape ``(FRAGMENT_COUNT * nat,)``, and positions,
        shape ``(FRAGMENT_COUNT * nat, 3)``.
    """
    fragment = mols[name]
    numbers = fragment["numbers"].repeat(FRAGMENT_COUNT).to(DEVICE)
    positions = fragment["positions"].to(**dd)

    offsets = FRAGMENT_SPACING * torch.arange(FRAGMENT_COUNT, **dd)
    shifts = torch.zeros((FRAGMENT_COUNT, 1, 3), **dd)
    shifts[:, 0, 0] = offsets

    shifted = positions.unsqueeze(0) + shifts
    return numbers, shifted.reshape(-1, 3)


def test_defaults_match_sdftd3() -> None:
    """The default cutoffs are the ones s-dftd3 itself defaults to."""
    # From `realspace_cutoff` in s-dftd3's src/dftd3/cutoff.f90 (v1.6.0):
    # cn_default = 40, disp2_default = 60, disp3_default = 40.
    assert defaults.D3_CN_CUTOFF == 40.0
    assert defaults.D3_DISP2_CUTOFF == 60.0
    assert defaults.D3_DISP3_CUTOFF == 40.0

    cutoff = Cutoff()
    assert float(cutoff.cn) == defaults.D3_CN_CUTOFF
    assert float(cutoff.disp2) == defaults.D3_DISP2_CUTOFF
    assert float(cutoff.disp3) == defaults.D3_DISP3_CUTOFF


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_chain_matches_reference(dtype: torch.dtype, name: str) -> None:
    """A chain of separated fragments matches s-dftd3 at the defaults."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # Tighter than the `sqrt(eps)` used elsewhere in this suite: the
    # smallest shift any cutoff causes here is 2.8e-10, so a looser
    # tolerance would let a wrong cutoff through in double precision.
    # Measured agreement in float64 is 1.1e-18, tolerance 2.2e-15.
    tol = 10 * torch.finfo(dtype).eps

    numbers, positions = fragment_chain(name, dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    # No cutoff is passed to the reference, so it uses s-dftd3's own.
    ref = reference_energy_per_atom(numbers, positions, par)
    energy = dftd3(numbers, positions, par)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == energy.cpu()


@pytest.mark.parametrize("name", ["H2O", "SiH4"])
@pytest.mark.parametrize("field,value", CHANGED_CUTOFFS)
def test_changed_cutoff_matches_reference(
    name: str, field: str, value: float
) -> None:
    """Each cutoff on its own is applied the way s-dftd3 applies it."""
    # float64 only: the shifts this test relies on are around 1e-10 Hartree
    # on an energy of about 1e-3, which is below float32's resolution.
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    numbers, positions = fragment_chain(name, dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    changed = Cutoff(**{field: value}, **dd)

    # Measured agreement is 1.1e-18 absolute, four orders below this.
    ref = reference_energy_per_atom(numbers, positions, par, cutoff=changed)
    energy = dftd3(numbers, positions, par, cutoff=changed)
    assert pytest.approx(ref.cpu(), abs=1e-14) == energy.cpu()

    # Only meaningful if this geometry notices the changed cutoff. The
    # threshold sits above the 1e-14 tolerance above and below the smallest
    # shift measured here (2.8e-10).
    energy_default = dftd3(numbers, positions, par)
    shift = float(torch.sum(energy - energy_default).abs())
    assert shift > 1e-11


def test_cutoff_rejects_non_scalar() -> None:
    """A cutoff has to be a single radius, not one per atom."""
    with pytest.raises(ValueError):
        Cutoff(disp2=torch.tensor([50.0, 60.0]))


def test_cutoff_is_cast_to_the_calculation() -> None:
    """A cutoff given in another dtype is cast, not rejected."""
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    numbers, positions = fragment_chain("H2O", dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    single = Cutoff(dtype=torch.float32)
    double = Cutoff(**dd)

    assert single.dtype == torch.float32
    energy = dftd3(numbers, positions, par, cutoff=single)
    expected = dftd3(numbers, positions, par, cutoff=double)

    assert energy.dtype == torch.float64
    assert pytest.approx(expected.cpu()) == energy.cpu()


def test_dftd3_rejects_a_single_cutoff() -> None:
    """One cutoff for everything is ambiguous and has to be rejected."""
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    numbers, positions = fragment_chain("H2O", dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    with pytest.raises(TypeError):
        dftd3(numbers, positions, par, cutoff=torch.tensor(50.0, **dd))


@pytest.mark.large
def test_large_molecule_matches_reference() -> None:
    """A molecule far larger than every cutoff matches s-dftd3 exactly.

    Uses about 2 GB: the three-body term builds several dense
    ``(nat, nat, nat)`` tensors and this molecule has 251 atoms.
    """
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    path = Path(__file__).parents[1] / "molecules" / "c83h168.xyz"
    numbers, positions = read(path, **dd)
    numbers = numbers.to(DEVICE)

    par = {k: v.to(**dd) for k, v in param.items()}

    # No cutoff for the reference, so it uses s-dftd3's own. All three
    # matter here: moving `cn` to 25, `disp2` to 50 or `disp3` to 50 shifts
    # this energy by 1e-7 or more, against a measured agreement of 2.2e-16.
    ref = reference_energy_per_atom(numbers, positions, par)
    energy = dftd3(numbers, positions, par)

    assert pytest.approx(ref.cpu(), abs=1e-14) == energy.cpu()
