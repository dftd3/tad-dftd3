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

from tad_dftd3 import damping, data, disp, utils
from tad_dftd3._typing import DD

from .conftest import DEVICE
from .samples import samples

sample_list = ["AmF3", "SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]

# TPSS0-D3BJ-ATM parameters
param = {
    "s6": torch.tensor(1.0000),
    "s8": torch.tensor(1.2576),
    "s9": torch.tensor(1.0000),
    "alp": torch.tensor(14.00),
    "a1": torch.tensor(0.3768),
    "a2": torch.tensor(4.5865),
}

# TPSS0-D3BJ parameters
param_noatm = {k: torch.tensor(0.0) if k == "s9" else v for k, v in param.items()}


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c6 = samples["PbH4-BiH3"]["c6"]

    # r4r2 wrong shape
    with pytest.raises(ValueError):
        r4r2 = torch.tensor([1.0])
        disp.dispersion(numbers, positions, param, c6, r4r2=r4r2)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        disp.dispersion(numbers, positions, param, c6)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_disp2_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["disp2"].to(**dd)
    c6 = sample["c6"].to(**dd)
    rvdw = data.vdw_rad_d3.to(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = data.sqrt_z_r4_over_r2.to(**dd)[numbers]
    cutoff = torch.tensor(50.0, **dd)

    par = {k: v.to(**dd) for k, v in param_noatm.items()}

    energy = disp.dispersion(
        numbers,
        positions,
        par,
        c6,
        rvdw,
        r4r2,
        disp.rational_damping,
        cutoff=cutoff,
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", ["SiH4"])
def test_disp2_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = samples[name1], samples[name2]
    numbers = utils.pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = utils.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    c6 = utils.pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )
    ref = utils.pack(
        [
            sample1["disp2"].to(**dd),
            sample2["disp2"].to(**dd),
        ]
    )

    par = {k: v.to(**dd) for k, v in param_noatm.items()}

    energy = disp.dispersion(numbers, positions, par, c6)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_atm_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    c6 = sample["c6"].to(**dd)
    ref = sample["disp3"].to(**dd)

    rvdw = data.vdw_rad_d3.to(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]

    par = {k: v.to(**dd) for k, v in param.items()}

    energy = damping.dispersion_atm(
        numbers,
        positions,
        c6,
        rvdw,
        cutoff=torch.tensor(50.0, **dd),
        s9=par["s9"],
        alp=par["alp"],
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", ["SiH4"])
def test_atm_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample1, sample2 = samples[name1], samples[name2]
    numbers = utils.pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = utils.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    c6 = utils.pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )
    ref = utils.pack(
        [
            sample1["disp3"].to(**dd),
            sample2["disp3"].to(**dd),
        ]
    )

    par = {k: v.to(**dd) for k, v in param.items()}

    rvdw = data.vdw_rad_d3.to(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]

    energy = damping.dispersion_atm(
        numbers,
        positions,
        c6,
        rvdw,
        cutoff=torch.tensor(50.0, **dd),
        s9=par["s9"],
        alp=par["alp"],
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_full_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    c6 = sample["c6"].to(**dd)
    ref = (sample["disp2"] + sample["disp3"]).to(**dd)

    par = {k: v.to(**dd) for k, v in param.items()}

    energy = disp.dispersion(numbers, positions, par, c6)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()
