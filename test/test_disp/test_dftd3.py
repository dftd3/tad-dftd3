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
from tad_mctc.io.structure import pack_structures
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import Cutoff, damping, data, dftd3, model, reference
from tad_dftd3.data.table import TABLES, element_table
from tad_dftd3.ncoord import exp_count

from ..conftest import DEVICE
from ..reference import reference_energy_per_atom
from ..utils import approx_ref, load_structure


def _approx(expected: torch.Tensor, dtype: torch.dtype) -> object:
    """
    The tolerance against s-dftd3: tight in float64; in float32, the default
    of `pytest.approx` (relative 1e-6), which is already close to the
    precision of the dtype (measured: 3.8e-7).
    """
    if dtype == torch.float64:
        return approx_ref(expected, dtype)
    return pytest.approx(expected)


def test_fail() -> None:
    # On the CPU, also under `--cuda`: on CUDA, the lookup of an element
    # beyond the tables is a device-side assert, which cannot be caught and
    # breaks the device for every later test.
    cpu = torch.device("cpu")
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=cpu)

    # TPSS0-D3BJ-ATM parameters
    param = {
        "s6": torch.tensor(1.0000, device=cpu),
        "s8": torch.tensor(1.2576, device=cpu),
        "s9": torch.tensor(1.0000, device=cpu),
        "alp": torch.tensor(14.00, device=cpu),
        "a1": torch.tensor(0.3768, device=cpu),
        "a2": torch.tensor(4.5865, device=cpu),
    }

    # unsupported element, beyond the tables
    with pytest.raises(IndexError):
        dftd3(
            Structure(
                numbers=torch.tensor([1, 105], device=cpu), positions=positions
            ),
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
    param = {
        "s8": torch.tensor(1.0),
        "a1": torch.tensor(0.3768),
        "a2": torch.tensor(4.5865),
    }

    tables = {
        "rcov": radii.COV_D3(dtype=torch.float, device=DEVICE),
        "rvdw": radii.VDW_PAIRWISE(dtype=torch.float, device=DEVICE),
        "r4r2": data.R4R2(dtype=torch.float, device=DEVICE),
    }
    per_atom = {
        "rcov": tables["rcov"][numbers],
        "rvdw": tables["rvdw"][numbers.unsqueeze(-1), numbers.unsqueeze(-2)],
        "r4r2": tables["r4r2"][numbers],
    }
    structure = Structure(numbers=numbers, positions=positions)
    with pytest.raises(ValueError, match=f"{name}_table"):
        dftd3(
            structure,
            param,
            **{f"{name}_table": per_atom[name]},  # type: ignore[arg-type]
        )

    # batched and truncated tables are rejected as well
    table = tables[name]
    for t in (table.expand(2, *table.shape), table[..., :-1]):
        with pytest.raises(ValueError, match=f"{name}_table"):
            dftd3(
                structure,
                param,
                **{f"{name}_table": t},  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "source",
    [("mb16_43", "LiH"), ("mb16_43", "SiH4"), ("heavy28", "pbh4_bih3")],
)
def test_single(dtype: torch.dtype, source: tuple[str, str]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    structure = load_structure(*source, dd)

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

    ref = reference_energy_per_atom(
        load_structure(*source, {"device": DEVICE, "dtype": torch.double}),
        param,
        cutoff=cutoff,
    ).to(**dd)

    energy = dftd3(
        structure,
        param,
        ref=reference.Reference.load(**dd),
        rcov_table=rcov,
        rvdw_table=rvdw,
        r4r2_table=r4r2,
        cutoff=cutoff,
        counting_function=exp_count,
        weighting_function=model.gaussian_log_weight,
    )

    assert energy.dtype == dtype
    assert _approx(ref.cpu(), dtype) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sources = [("heavy28", "pbh4_bih3"), ("other", "C6H5I-CH3SH")]
    structure = pack_structures([load_structure(*src, dd) for src in sources])
    param = {
        "s6": torch.tensor(1.0000, **dd),
        "s8": torch.tensor(1.2576, **dd),
        "s9": torch.tensor(0.0000, **dd),  # no ATM!
        "alp": torch.tensor(14.00, **dd),
        "a1": torch.tensor(0.3768, **dd),
        "a2": torch.tensor(4.5865, **dd),
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
    assert _approx(ref.cpu(), dtype) == energy.cpu()


def test_default_tables_cached() -> None:
    """The default tables are built once per device and dtype, not per call."""
    from tad_mctc.data.table import _TABLE_CACHE

    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    param = {
        "s8": torch.tensor(1.0),
        "s9": torch.tensor(1.0),
        "a1": torch.tensor(0.3768),
        "a2": torch.tensor(4.5865),
    }
    defaults = (radii.COV_D3, radii.VDW_PAIRWISE, data.R4R2)

    _TABLE_CACHE.clear()
    structure = Structure(numbers=numbers, positions=positions)
    dftd3(structure, param)
    tables = {fn: dict(_TABLE_CACHE[fn]) for fn in defaults}
    assert all(len(per_table) == 1 for per_table in tables.values())

    dftd3(structure, param)
    for fn, per_table in tables.items():
        for key, cached in per_table.items():
            assert _TABLE_CACHE[fn][key] is cached


def test_table_shapes() -> None:
    """The required shapes are those of the default tables."""
    for name, (default, shape) in TABLES.items():
        assert tuple(default().shape) == shape, name  # type: ignore[call-arg]


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
    param = {
        "s8": torch.tensor(1.0, **dd),
        "a1": torch.tensor(0.4, **dd),
        "a2": torch.tensor(4.6, **dd),
    }

    ref = reference.Reference.load(**dd)
    structure = Structure(numbers=numbers, positions=positions)

    def manual(cn: Tensor) -> Tensor:
        weights = model.weight_references(numbers, cn, ref)
        c6 = model.atomic_c6(numbers, weights, ref)
        return disp.dispersion(structure, param, c6)

    cn_model = ncoord.cn_d3.replace(cutoff=defaults.D3_CN_CUTOFF)
    energy = dftd3(structure, param)

    assert (
        pytest.approx(energy.cpu(), abs=1e-14)
        == manual(cn_model(structure)).cpu()
    )

    # only meaningful if the default cutoff of `cn_d3` changes the energy
    assert (
        pytest.approx(energy.cpu(), abs=1e-14)
        != manual(ncoord.cn_d3(structure)).cpu()
    )
