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
Test calculation of two-body and three-body dispersion terms.
"""
from math import sqrt

import pytest
import torch

from tad_dftd3 import damping, data, disp, util

from .samples import samples


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    param = {
        "a1": torch.tensor(0.49484001),
        "s8": torch.tensor(0.78981345),
        "a2": torch.tensor(5.73083694),
    }
    c6 = samples["PbH4-BiH3"]["c6"]

    # r4r2 wrong shape
    with pytest.raises(ValueError):
        r4r2 = torch.tensor([1.0])
        disp.dispersion(numbers, positions, param, c6, r4r2=r4r2)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        disp.dispersion(numbers, positions, param, c6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp2_single(dtype: torch.dtype) -> None:
    sample = samples["PbH4-BiH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["disp2"].type(dtype)
    c6 = sample["c6"].type(dtype)
    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = data.sqrt_z_r4_over_r2[numbers]
    cutoff = torch.tensor(50.0, dtype=dtype)

    param = {
        "a1": positions.new_tensor(0.49484001),
        "s8": positions.new_tensor(0.78981345),
        "a2": positions.new_tensor(5.73083694),
    }

    energy = disp.dispersion(
        numbers, positions, param, c6, rvdw, r4r2, disp.rational_damping, cutoff=cutoff
    )

    assert energy.dtype == dtype
    assert pytest.approx(energy) == ref


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disp2_batch(dtype: torch.dtype) -> None:
    sample1, sample2 = samples["PbH4-BiH3"], samples["C6H5I-CH3SH"]
    numbers = util.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = util.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    c6 = util.pack(
        (
            sample1["c6"].type(dtype),
            sample2["c6"].type(dtype),
        )
    )
    ref = util.pack(
        (
            sample1["disp2"].type(dtype),
            sample2["disp2"].type(dtype),
        )
    )

    param = {
        "a1": positions.new_tensor(0.49484001),
        "s8": positions.new_tensor(0.78981345),
        "a2": positions.new_tensor(5.73083694),
    }

    energy = disp.dispersion(numbers, positions, param, c6)

    assert energy.dtype == dtype
    assert pytest.approx(energy) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["SiH4", "MB16_43_01"])
def test_atm_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    c6 = sample["c6"].type(dtype)
    ref = (sample["disp3"] - sample["disp2"]).type(dtype)

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.2576),
        "s9": positions.new_tensor(1.0),
        "alp": positions.new_tensor(14.0),
        "a1": positions.new_tensor(0.3768),
        "a2": positions.new_tensor(4.5865),
    }

    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(dtype)

    energy = damping.dispersion_atm(
        numbers,
        positions,
        c6,
        rvdw,
        cutoff=positions.new_tensor(50.0),
        s9=param["s9"],
        alp=param["alp"],
    )

    assert energy.dtype == dtype
    assert pytest.approx(energy, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["SiH4", "MB16_43_01"])
@pytest.mark.parametrize("name2", ["SiH4"])
def test_atm_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
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
    c6 = util.pack(
        [
            sample1["c6"].type(dtype),
            sample2["c6"].type(dtype),
        ]
    )
    ref = util.pack(
        [
            (sample1["disp3"] - sample1["disp2"]).type(dtype),
            (sample2["disp3"] - sample2["disp2"]).type(dtype),
        ]
    )

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.2576),
        "s9": positions.new_tensor(1.0),
        "alp": positions.new_tensor(14.0),
        "a1": positions.new_tensor(0.3768),
        "a2": positions.new_tensor(4.5865),
    }

    rvdw = data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)].type(dtype)

    energy = damping.dispersion_atm(
        numbers,
        positions,
        c6,
        rvdw,
        cutoff=positions.new_tensor(50.0),
        s9=param["s9"],
        alp=param["alp"],
    )

    assert energy.dtype == dtype
    assert pytest.approx(energy, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["SiH4", "MB16_43_01"])
def test_full_single(dtype: torch.dtype, name: str) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    c6 = sample["c6"].type(dtype)
    ref = sample["disp3"].type(dtype)

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.2576),
        "s9": positions.new_tensor(1.0),
        "alp": positions.new_tensor(14.0),
        "a1": positions.new_tensor(0.3768),
        "a2": positions.new_tensor(4.5865),
    }

    energy = disp.dispersion(numbers, positions, param, c6)

    assert energy.dtype == dtype
    assert pytest.approx(energy, abs=tol) == ref
