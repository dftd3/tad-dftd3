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
from tad_mctc import Structure
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.typing import DD

from tad_dftd3 import Cutoff, damping, data, disp

from ..conftest import DEVICE
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
param_noatm = {
    k: torch.tensor(0.0) if k == "s9" else v for k, v in param.items()
}


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c6 = samples["PbH4-BiH3"]["c6"]

    # wrong numbers, rejected by `Structure` itself
    with pytest.raises(RuntimeError):
        Structure(numbers=torch.tensor([1]), positions=positions)

    # unsupported element
    with pytest.raises(ValueError):
        disp.dispersion(
            Structure(numbers=torch.tensor([1, 105]), positions=positions),
            param,
            c6,
        )


@pytest.mark.parametrize("name", ["rvdw_table", "r4r2_table"])
def test_fail_table(name: str) -> None:
    """Per-atom (or per-pair) values and truncated tables are rejected."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c6 = torch.ones((2, 2))

    table = {"rvdw_table": radii.VDW_PAIRWISE, "r4r2_table": data.R4R2}[name](
        dtype=torch.float
    )
    wrong = {
        "per atom": (
            table[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
            if name == "rvdw_table"
            else table[numbers]
        ),
        "truncated": table[..., :-1],
        "batched": table.expand(2, *table.shape),
    }
    for t in wrong.values():
        with pytest.raises(ValueError, match=name):
            disp.dispersion(
                Structure(numbers=numbers, positions=positions),
                param,
                c6,
                **{name: t}
            )


def test_fail_renamed_and_positional() -> None:
    """
    The per-atom arguments of 0.7.0 fail loudly in every function, whether
    passed by their old name or by position.
    """
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    structure = Structure(numbers=numbers, positions=positions)
    c6 = torch.ones((2, 2))
    r4r2 = data.R4R2(dtype=torch.float)[numbers]
    rvdw = radii.VDW_PAIRWISE(dtype=torch.float)[
        numbers.unsqueeze(-1), numbers.unsqueeze(-2)
    ]

    for func in (disp.dispersion, disp.dispersion2):
        with pytest.raises(TypeError, match="r4r2_table"):
            func(structure, param, c6, r4r2=r4r2)
    for func in (disp.dispersion, disp.dispersion3):
        with pytest.raises(TypeError, match="rvdw_table"):
            func(structure, param, c6, rvdw=rvdw)
    with pytest.raises(TypeError, match="rvdw_table"):
        damping.dispersion_atm(structure, c6, rvdw=rvdw)

    # positional calls of 0.7.0: everything after `c6` is keyword-only
    with pytest.raises(TypeError, match="positional"):
        disp.dispersion(
            structure,
            param,
            c6,
            rvdw,
            None,
        )
    with pytest.raises(TypeError, match="positional"):
        disp.dispersion2(
            structure,
            param,
            c6,
            r4r2,
            disp.rational_damping,
            50.0,
        )
    with pytest.raises(TypeError, match="positional"):
        disp.dispersion3(
            structure,
            param,
            c6,
            rvdw,
            50.0,
        )
    with pytest.raises(TypeError, match="positional"):
        damping.dispersion_atm(structure, c6, rvdw, 50.0)


def test_fail_numbers_positions() -> None:
    """
    The calls of 0.7.0, with `numbers` and `positions` instead of a
    `Structure`, fail with a hint at the new call in every function.
    """
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c6 = torch.ones((2, 2))

    calls = [
        lambda: disp.dftd3(numbers, positions, param),
        lambda: disp.dispersion(numbers, positions, param, c6),
        lambda: disp.dispersion2(numbers, positions, param, c6),
        lambda: disp.dispersion3(numbers, positions, param, c6),
        lambda: damping.dispersion_atm(numbers, positions, c6),
        lambda: disp.dftd3(structure=numbers, param=param),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="Structure"):
            call()


def test_float_s9() -> None:
    """`s9`, `rs9` and `alp` may be Python numbers."""
    dd: DD = {"device": DEVICE, "dtype": torch.double}
    sample = samples["SiH4"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    c6 = sample["c6"].to(**dd)

    par = {k: v.to(**dd) for k, v in param.items()}
    structure = Structure(numbers=numbers, positions=positions)
    ref = disp.dispersion(structure, par, c6)

    par_float = {**par, "s9": 1.0, "alp": 14.0}
    energy = disp.dispersion(structure, par_float, c6)
    assert pytest.approx(ref.cpu(), abs=1e-14) == energy.cpu()

    atm = damping.dispersion_atm(
        structure,
        c6,
        s9=1.0,
        rs9=4.0 / 3.0,
        alp=14.0,
    )
    atm_ref = damping.dispersion_atm(
        structure,
        c6,
        s9=par["s9"],
        alp=par["alp"],
    )
    assert pytest.approx(atm_ref.cpu(), abs=1e-14) == atm.cpu()

    # `s9 = 0.0` skips the three-body term
    no_atm = disp.dispersion(structure, {**par, "s9": 0.0}, c6)
    assert pytest.approx(ref.cpu() - atm_ref.cpu(), abs=1e-14) == no_atm.cpu()


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
    rvdw = radii.VDW_PAIRWISE(**dd)
    r4r2 = data.R4R2(**dd)
    cutoff = Cutoff(disp2=50.0)

    par = {k: v.to(**dd) for k, v in param_noatm.items()}

    energy = disp.dispersion(
        Structure(numbers=numbers, positions=positions),
        par,
        c6,
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        damping_function=disp.rational_damping,
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
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    c6 = pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["disp2"].to(**dd),
            sample2["disp2"].to(**dd),
        ]
    )

    par = {k: v.to(**dd) for k, v in param_noatm.items()}

    energy = disp.dispersion(
        Structure(numbers=numbers, positions=positions), par, c6
    )

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

    rvdw = radii.VDW_PAIRWISE(**dd)

    par = {k: v.to(**dd) for k, v in param.items()}

    energy = damping.dispersion_atm(
        Structure(numbers=numbers, positions=positions),
        c6,
        rvdw_table=rvdw,
        cutoff=50.0,
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
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    c6 = pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )
    ref = pack(
        [
            sample1["disp3"].to(**dd),
            sample2["disp3"].to(**dd),
        ]
    )

    par = {k: v.to(**dd) for k, v in param.items()}

    rvdw = radii.VDW_PAIRWISE(**dd)

    energy = damping.dispersion_atm(
        Structure(numbers=numbers, positions=positions),
        c6,
        rvdw_table=rvdw,
        cutoff=50.0,
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

    energy = disp.dispersion(
        Structure(numbers=numbers, positions=positions), par, c6
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()
