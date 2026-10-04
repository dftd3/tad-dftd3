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
from tad_mctc import Structure
from tad_mctc.batch import pack
from tad_mctc.data import radii
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import Cutoff, damping, data, dftd3, model, reference
from tad_dftd3.data.table import TABLES, element_table
from tad_dftd3.ncoord import exp_count

from ..conftest import DEVICE
from .samples import samples


def test_fail() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": torch.tensor(1.0000),
        "s8": torch.tensor(1.2576),
        "s9": torch.tensor(1.0000),
        "alp": torch.tensor(14.00),
        "a1": torch.tensor(0.3768),
        "a2": torch.tensor(4.5865),
    }

    # unsupported element
    with pytest.raises(ValueError):
        dftd3(
            Structure(numbers=torch.tensor([1, 105]), positions=positions),
            param,
        )

    # wrong numbers, rejected by `Structure` itself
    with pytest.raises(RuntimeError):
        Structure(numbers=torch.tensor([1]), positions=positions)


@pytest.mark.parametrize("name", ["rcov", "rvdw", "r4r2"])
def test_fail_per_atom(name: str) -> None:
    """Per-atom (or per-pair) values are rejected, not indexed again."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    param = {"a1": torch.tensor(0.3768), "a2": torch.tensor(4.5865)}

    tables = {
        "rcov": radii.COV_D3(dtype=torch.float),
        "rvdw": radii.VDW_PAIRWISE(dtype=torch.float),
        "r4r2": data.R4R2(dtype=torch.float),
    }
    per_atom = {
        "rcov": tables["rcov"][numbers],
        "rvdw": tables["rvdw"][numbers.unsqueeze(-1), numbers.unsqueeze(-2)],
        "r4r2": tables["r4r2"][numbers],
    }
    with pytest.raises(ValueError, match=f"{name}_table"):
        dftd3(
            Structure(numbers=numbers, positions=positions),
            param,
            **{f"{name}_table": per_atom[name]},
        )

    # batched and truncated tables are rejected as well
    table = tables[name]
    for t in (table.expand(2, *table.shape), table[..., :-1]):
        with pytest.raises(ValueError, match=f"{name}_table"):
            dftd3(
                Structure(numbers=numbers, positions=positions),
                param,
                **{f"{name}_table": t},
            )


@pytest.mark.parametrize("name", ["rcov", "rvdw", "r4r2"])
def test_fail_renamed(name: str) -> None:
    """
    The names of 0.7.0, which took per-atom values, are rejected outright,
    also for the per-atom values of a system with as many atoms as the
    table has entries, whose shape cannot tell them apart from a table.
    """
    nat = 104 if name == "rvdw" else 119
    numbers = torch.ones(nat, dtype=torch.long)
    positions = torch.rand((nat, 3)) * 20.0
    param = {"a1": torch.tensor(0.3768), "a2": torch.tensor(4.5865)}

    table = {
        "rcov": radii.COV_D3,
        "rvdw": radii.VDW_PAIRWISE,
        "r4r2": data.R4R2,
    }[name](dtype=torch.float)
    per_atom = (
        table[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
        if name == "rvdw"
        else table[numbers]
    )

    with pytest.raises(TypeError, match=f"'{name}_table'"):
        dftd3(
            Structure(numbers=numbers, positions=positions),
            param,
            **{name: per_atom},
        )

    # also when the old name is given the table itself
    with pytest.raises(TypeError, match=f"'{name}_table'"):
        dftd3(
            Structure(numbers=numbers, positions=positions),
            param,
            **{name: table},
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", ["LiH", "SiH4", "PbH4-BiH3"])
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = (sample["disp2"] + sample["disp3"]).to(**dd)

    rcov = radii.COV_D3(**dd)
    rvdw = radii.VDW_PAIRWISE(**dd)
    r4r2 = data.R4R2(**dd)
    cutoff = Cutoff(disp2=50.0, disp3=50.0)

    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(1.2576, **dd),
        "s9": torch.tensor(1.0000, **dd),
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.3768, **dd),
        "a2": torch.tensor(4.5865, **dd),
    }

    energy = dftd3(
        Structure(numbers=numbers, positions=positions),
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
    ref = pack(
        (
            sample1["disp2"].to(**dd),
            sample2["disp2"].to(**dd),
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

    energy = dftd3(Structure(numbers=numbers, positions=positions), param)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu()) == energy.cpu()


def test_default_tables_cached() -> None:
    """The default tables are built once per device and dtype, not per call."""
    from tad_mctc.data.table import _TABLE_CACHE

    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    param = {
        "s9": torch.tensor(1.0),
        "a1": torch.tensor(0.3768),
        "a2": torch.tensor(4.5865),
    }
    defaults = (radii.COV_D3, radii.VDW_PAIRWISE, data.R4R2)

    _TABLE_CACHE.clear()
    dftd3(Structure(numbers=numbers, positions=positions), param)
    tables = {fn: dict(_TABLE_CACHE[fn]) for fn in defaults}
    assert all(len(per_table) == 1 for per_table in tables.values())

    dftd3(Structure(numbers=numbers, positions=positions), param)
    for fn, per_table in tables.items():
        for key, cached in per_table.items():
            assert _TABLE_CACHE[fn][key] is cached


def test_table_shapes() -> None:
    """The required shapes are those of the default tables."""
    for name, (default, shape) in TABLES.items():
        assert tuple(default().shape) == shape, name


def test_given_table_skips_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table that is passed is checked without building the default."""

    def default(**_: object) -> Tensor:
        raise AssertionError("default table built")

    monkeypatch.setitem(TABLES, "r4r2_table", (default, (119,)))

    table = data.R4R2(dtype=torch.double)
    like = torch.zeros(1, dtype=torch.double)

    assert element_table(table, "r4r2_table", like) is table

    with pytest.raises(ValueError, match="r4r2_table"):
        element_table(table[:-1], "r4r2_table", like)


def test_manual_pipeline_matches_dftd3() -> None:
    """
    The step-by-step pipeline of the examples reproduces `dftd3` when two
    fragments are farther apart than the 25 Bohr default cutoff of
    `tad_mctc.ncoord.cn_d3`, but within the 40 Bohr cutoff of `dftd3`.
    """
    # pylint: disable=import-outside-toplevel
    from tad_mctc import Structure

    from tad_dftd3 import defaults, disp, ncoord

    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    numbers = torch.tensor([1, 1, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.4],
            [30.0, 0.0, 0.0],
            [30.0, 0.0, 1.4],
        ],
        **dd,
    )
    param = {"a1": torch.tensor(0.4, **dd), "a2": torch.tensor(4.6, **dd)}

    ref = reference.Reference(**dd)
    structure = Structure(numbers=numbers, positions=positions)

    def manual(cn: Tensor) -> Tensor:
        weights = model.weight_references(numbers, cn, ref)
        c6 = model.atomic_c6(numbers, weights, ref)
        return disp.dispersion(
            Structure(numbers=numbers, positions=positions), param, c6
        )

    cn_model = ncoord.cn_d3.replace(cutoff=defaults.D3_CN_CUTOFF)
    energy = dftd3(Structure(numbers=numbers, positions=positions), param)

    assert (
        pytest.approx(energy.cpu(), abs=1e-14)
        == manual(cn_model(structure)).cpu()
    )

    # only meaningful if the default cutoff of `cn_d3` changes the energy
    assert (
        pytest.approx(energy.cpu(), abs=1e-14)
        != manual(ncoord.cn_d3(structure)).cpu()
    )
