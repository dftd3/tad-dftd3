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
Test calculation of dispersion energy and nuclear gradients.
"""

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.data.molecules import mols as samples
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import damping, data, dftd3, model, reference
from tad_dftd3.cutoff import Cutoff
from tad_dftd3.ncoord import exp_count

from ..conftest import DEVICE
from ..reference import reference_energy_per_atom

# Several D3(BJ) parameter sets, so the sweep below is not calibrated to
# one functional's damping curve.
TPSS0_D3BJ_ATM = {
    "s6": torch.tensor(1.0000),
    "s8": torch.tensor(1.2576),
    "s9": torch.tensor(1.0000),
    "alp": torch.tensor(14.00),
    "a1": torch.tensor(0.3768),
    "a2": torch.tensor(4.5865),
}

PBE_D3BJ = {
    "s6": torch.tensor(1.0000),
    "s8": torch.tensor(0.7875),
    "a1": torch.tensor(0.4289),
    "a2": torch.tensor(4.4407),
}
"""No ``s9`` key, so no ATM term."""

PBE0_D3BJ_ATM = {
    "s6": torch.tensor(1.0000),
    "s8": torch.tensor(1.2177),
    "s9": torch.tensor(1.0000),
    "alp": torch.tensor(14.00),
    "a1": torch.tensor(0.4145),
    "a2": torch.tensor(4.8593),
}

B3LYP_D3BJ_ATM = {
    "s6": torch.tensor(1.0000),
    "s8": torch.tensor(1.9889),
    "s9": torch.tensor(1.0000),
    "alp": torch.tensor(14.00),
    "a1": torch.tensor(0.3981),
    "a2": torch.tensor(4.4211),
}

PARAM_IDS = ["TPSS0-ATM", "PBE", "PBE0-ATM", "B3LYP-ATM"]
PARAMS = [TPSS0_D3BJ_ATM, PBE_D3BJ, PBE0_D3BJ_ATM, B3LYP_D3BJ_ATM]

NAMES = [
    "LiH",
    "H",
    "Rn",
    "H2",
    "Li2",
    "S2",
    "H2O",
    "CO2",
    "NH3",
    "NH3-dimer",
    "CH4",
    "SiH4",
    "MB16_43_01",
    "MB16_43_02",
    "MB16_43_03",
    "MB16_43_07",
    "MB16_43_08",
    "PbH4-BiH3",
    "C6H5I-CH3SH",
    "Al3+Ar6",
    "ZnOOH-",
    "AmF3",
    "actinides",
    "H2-stretched",
    "H2-compressed",
    "MB16_43_01-perturbed",
    "C6H5I-CH3SH-perturbed",
]
"""Geometries for :func:`test_single` below. Mostly ``mstore`` samples,
chosen to also cover a single atom (``"H"``, ``"Rn"``), a linear molecule
(every diatomic here, plus ``"CO2"``) and elements above Z = 86
(``"AmF3"``, Z = 95; ``"actinides"``, Z = 87-103). The last four are built
by :func:`_geometry_for` rather than looked up: a hydrogen pair stretched
and compressed past what any equilibrium geometry visits, and two
``mstore`` geometries displaced by noise, so the sweep is not limited to
positions a sample happens to sit at."""

STRETCHED_PAIR_DISTANCE = 15.0
"""Bohr. Past the equilibrium separation, where damping rather than the
C6 term controls how fast the energy vanishes."""

COMPRESSED_PAIR_DISTANCE = 0.5
"""Bohr. Well inside the van der Waals radius, the opposite limit of BJ
damping from :data:`STRETCHED_PAIR_DISTANCE`."""

PERTURBATION_SIGMA = 0.05
"""Bohr. Per-coordinate size of the noise added to a perturbed geometry."""


def _geometry_for(name: str, dd: DD) -> tuple[Tensor, Tensor]:
    """Numbers and positions for one entry of :data:`NAMES`."""
    if name == "H2-stretched":
        numbers = torch.tensor([1, 1], device=dd["device"])
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [STRETCHED_PAIR_DISTANCE, 0.0, 0.0]], **dd
        )
        return numbers, positions

    if name == "H2-compressed":
        numbers = torch.tensor([1, 1], device=dd["device"])
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [COMPRESSED_PAIR_DISTANCE, 0.0, 0.0]], **dd
        )
        return numbers, positions

    if name.endswith("-perturbed"):
        base = samples[name.removesuffix("-perturbed")]
        numbers = base["numbers"].to(DEVICE)
        positions = base["positions"].to(**dd)

        # Fixed seed so the sweep is reproducible run to run.
        generator = torch.Generator().manual_seed(0)
        noise = PERTURBATION_SIGMA * torch.randn(
            positions.shape, generator=generator, dtype=positions.dtype
        )
        return numbers, positions + noise.to(positions.device)

    sample = samples[name]
    return sample["numbers"].to(DEVICE), sample["positions"].to(**dd)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("param", PARAMS, ids=PARAM_IDS)
@pytest.mark.parametrize("name", NAMES)
def test_single(
    dtype: torch.dtype, param: dict[str, Tensor], name: str
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = _geometry_for(name, dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    # No cutoff for the reference, so it uses s-dftd3's own; the explicit
    # `Cutoff()` below has to reproduce those.
    ref = reference_energy_per_atom(numbers, positions, par)

    rcov = radii.COV_D3(**dd)[numbers]
    rvdw = radii.VDW_PAIRWISE(**dd)[
        numbers.unsqueeze(-1), numbers.unsqueeze(-2)
    ]
    r4r2 = data.R4R2(**dd)[numbers]

    energy = dftd3(
        numbers,
        positions,
        par,
        ref=reference.Reference(**dd),
        rcov=rcov,
        rvdw=rvdw,
        r4r2=r4r2,
        cutoff=Cutoff(**dd),
        counting_function=exp_count,
        weighting_function=model.gaussian_weight,
        damping_function=damping.rational_damping,
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = (samples["PbH4-BiH3"], samples["C6H5I-CH3SH"])
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(1.2576, **dd),
        "s9": torch.tensor(0.0000, **dd),  # no ATM!
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.3768, **dd),
        "a2": torch.tensor(4.5865, **dd),
    }

    # s-dftd3 has no notion of a batch of independent molecules; compute the
    # reference for each molecule on its own and pack them the same way the
    # inputs above were packed.
    ref = pack(
        (
            reference_energy_per_atom(
                sample1["numbers"].to(DEVICE),
                sample1["positions"].to(**dd),
                param,
            ),
            reference_energy_per_atom(
                sample2["numbers"].to(DEVICE),
                sample2["positions"].to(**dd),
                param,
            ),
        )
    )

    energy = dftd3(numbers, positions, param)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()


def test_fail() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # unsupported element
    with pytest.raises(ValueError):
        dftd3(torch.tensor([1, 105]), positions, TPSS0_D3BJ_ATM)

    # wrong numbers
    with pytest.raises(ValueError):
        dftd3(torch.tensor([1]), positions, TPSS0_D3BJ_ATM)
