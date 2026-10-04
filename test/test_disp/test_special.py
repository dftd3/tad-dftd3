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
Test calculation of dispersion energy for a system, which fail without the
weird handling of exceptional values in the calculation of the weights.
"""

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.io.structure import pack_structures
from tad_mctc.data import radii
from tad_mctc.typing import DD

from tad_dftd3 import Cutoff, damping, data, dftd3, model, reference
from tad_dftd3.ncoord import exp_count

from ..conftest import DEVICE
from ..reference import reference_energy_per_atom
from ..utils import load_structure, ref_tol


def _tol(dtype: torch.dtype) -> dict[str, float]:
    """Tight in float64; the default of `pytest.approx` in float32."""
    return ref_tol(dtype) if dtype == torch.float64 else {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source", [("other", "La3N@C80")])
def test_single(dtype: torch.dtype, source: tuple[str, str]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_structure(*source, dd)

    rcov = radii.COV_D3(**dd)
    rvdw = radii.VDW_PAIRWISE(**dd)
    r4r2 = data.R4R2(**dd)
    cutoff = Cutoff(disp2=50.0, disp3=50.0)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(2.4000, **dd),
        "s9": torch.tensor(0.0000, **dd),
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.6300, **dd),
        "a2": torch.tensor(5.0000, **dd),
    }

    ref = reference_energy_per_atom(
        load_structure(*source, {"device": DEVICE, "dtype": torch.double}),
        param,
        cutoff=cutoff,
    ).to(**dd)

    energy = dftd3(
        structure,
        param,
        ref=reference.Reference(**dd),
        rcov_table=rcov,
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        cutoff=cutoff,
        counting_function=exp_count,
        weighting_function=model.gaussian_weight,
        damping_function=damping.rational_damping,
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), **_tol(dtype)) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sources = [("mb16_43", "LiH"), ("other", "La3N@C80")]
    structure = pack_structures([load_structure(*src, dd) for src in sources])
    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(2.4000, **dd),
        "s9": torch.tensor(0.0000, **dd),
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.6300, **dd),
        "a2": torch.tensor(5.0000, **dd),
    }

    ref = pack(
        [
            reference_energy_per_atom(
                load_structure(*src, {"device": DEVICE, "dtype": torch.double}),
                param,
            ).to(**dd)
            for src in sources
        ]
    )

    energy = dftd3(structure, param)
    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), **_tol(dtype)) == energy.cpu()
