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
Test lookup of D3 damping parameters by functional name.
"""

from __future__ import annotations

import pytest
import torch

from tad_dftd3.param import get_functional_params

from ..conftest import DEVICE


def test_canonical_tpss0_bj() -> None:
    # TPSS0-D3BJ-ATM parameters, matching test/test_disp/test_dftd3.py::test_fail
    param = get_functional_params("tpss0", damping="bj", device=DEVICE)

    assert param["a1"] == pytest.approx(0.3768)
    assert param["s8"] == pytest.approx(1.2576)
    assert param["a2"] == pytest.approx(4.5865)
    assert param["s6"] == pytest.approx(1.0)
    assert param["s9"] == pytest.approx(1.0)
    assert param["alp"] == pytest.approx(14.0)


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("b3lyp5", "b3lyp"),
        ("b3-lyp5", "b3lyp"),
        ("B3LYP", "b3lyp"),
        ("pbeh", "pbe0"),
        ("bp86", "bp"),
        ("b88b95", "b1b95"),
    ],
)
def test_alias_resolution(alias: str, canonical: str) -> None:
    got = get_functional_params(alias, damping="bj")
    want = get_functional_params(canonical, damping="bj")

    assert got.keys() == want.keys()
    for key in got:
        assert got[key] == want[key]


def test_unknown_functional_raises() -> None:
    with pytest.raises(KeyError):
        get_functional_params("not-a-real-functional")


def test_missing_damping_variant_raises() -> None:
    # slaterdirac only has a "zero" entry, no "bj"
    with pytest.raises(ValueError):
        get_functional_params("slaterdirac", damping="bj")


def test_damping_preference_list_falls_through() -> None:
    # slaterdirac only has a "zero" entry, no "bj"
    got = get_functional_params("slaterdirac", damping=["bj", "zero"])
    want = get_functional_params("slaterdirac", damping="zero")

    assert got.keys() == want.keys()
    for key in got:
        assert got[key] == want[key]


def test_default_damping_preference_picks_bj() -> None:
    default = get_functional_params("b3lyp")
    bj = get_functional_params("b3lyp", damping="bj")

    assert default.keys() == bj.keys()
    for key in default:
        assert default[key] == bj[key]


def test_keep_meta() -> None:
    with_meta = get_functional_params("b3lyp", damping="bj", keep_meta=True)
    without_meta = get_functional_params("b3lyp", damping="bj", keep_meta=False)

    assert "doi" in with_meta
    assert "doi" not in without_meta


def test_returns_tensors() -> None:
    param = get_functional_params("b3lyp", damping="bj")
    for value in param.values():
        assert isinstance(value, torch.Tensor)
