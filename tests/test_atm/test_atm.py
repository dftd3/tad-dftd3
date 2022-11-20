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

from math import sqrt

import pytest
import torch

from tad_dftd3 import data, ncoord, model, damping, reference, util
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["SiH4", "MB16_43_01"])
def test_single(dtype: torch.dtype, name: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = (sample["energy_atm"] - sample["energy"]).type(dtype)

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": 1.0,
        "s8": 1.2576,
        "s9": 1.0,
        "alp": 14.0,
        "a1": 0.3768,
        "a2": 4.5865,
    }

    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(dtype)
    cutoff = torch.tensor(50.0, dtype=dtype)
    refmodel = reference.Reference().type(dtype)
    rcov = data.covalent_rad_d3[numbers].type(dtype)
    cn = ncoord.coordination_number(numbers, positions, rcov)
    weights = model.weight_references(numbers, cn, refmodel)
    c6 = model.atomic_c6(numbers, weights, refmodel)

    energy = damping.disp_atm(
        numbers,
        positions,
        c6,
        rvdw,
        cutoff,
        param.get("s9", 1.0),
        alp=param.get("alp", 14.0),
    )

    assert energy.dtype == dtype
    assert pytest.approx(energy, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4", "MB16_43_01"])
@pytest.mark.parametrize("name2", ["SiH4"])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    tol = sqrt(torch.finfo(dtype).eps) * 10
    sample1, sample2 = samples[name1], samples[name2]
    numbers = util.pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    positions = util.pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )
    ref = util.pack(
        [
            (sample1["energy_atm"] - sample1["energy"]).type(dtype),
            (sample2["energy_atm"] - sample2["energy"]).type(dtype),
        ]
    )

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": 1.0,
        "s8": 1.2576,
        "s9": 1.0,
        "alp": 14.0,
        "a1": 0.3768,
        "a2": 4.5865,
    }

    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(dtype)
    cutoff = torch.tensor(50.0, dtype=dtype)
    refmodel = reference.Reference().type(dtype)
    rcov = data.covalent_rad_d3[numbers].type(dtype)
    cn = ncoord.coordination_number(numbers, positions, rcov)
    weights = model.weight_references(numbers, cn, refmodel)
    c6 = model.atomic_c6(numbers, weights, refmodel)

    energy = damping.disp_atm(
        numbers,
        positions,
        c6,
        rvdw,
        cutoff,
        param.get("s9", 1.0),
        alp=param.get("alp", 14.0),
    )

    assert energy.dtype == dtype
    assert pytest.approx(energy, abs=tol) == ref
