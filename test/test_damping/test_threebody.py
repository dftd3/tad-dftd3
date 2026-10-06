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
The three-body damping is called by every evaluation of the ATM term.

s-dftd3 has a single three-body damping, so the comparisons against it
cannot tell whether a damping that is passed is used. Twice the zero damping
does, on every path. The dampings ported from dftd are tested in
``test_dftd_ports.py``.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.neighbor.list import build_neighborlist
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import Cutoff, D3Model, Damping, DampingParam, dftd3
from tad_dftd3.damping import (
    RationalTwoBody,
    TripleData,
    ZeroThreeBodyD3,
    dispersion_atm,
    dispersion_atm_periodic,
)
from tad_dftd3.sparse import dispersion_atm_sparse

from ..cells import cells
from ..conftest import DEVICE
from ..utils import load_structure

DD64: DD = {"device": DEVICE, "dtype": torch.double}

cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=10.0)

param = DampingParam(
    s8=torch.tensor(1.0, **DD64),
    a1=torch.tensor(0.4, **DD64),
    a2=torch.tensor(5.0, **DD64),
    s9=torch.tensor(1.0, **DD64),
)


class Doubled(ZeroThreeBodyD3):
    """Twice the zero damping."""

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        return 2.0 * super().__call__(triples, param)


def molecule() -> Structure:
    return load_structure("mb16_43", "SiH4", DD64)


def cell() -> Structure:
    return cells["urea"].to(DEVICE)


def _c6(structure: Structure) -> Tensor:
    return D3Model(cutoff=cutoff).c6(structure)


def test_dense_molecule() -> None:
    """The molecular evaluation calls the damping."""
    structure = molecule()
    c6 = _c6(structure)
    ref = dispersion_atm(structure, c6, param, cutoff=cutoff.disp3)
    out = dispersion_atm(
        structure, c6, param, damping=Doubled(), cutoff=cutoff.disp3
    )
    assert ref.abs().sum() > 0.0
    assert pytest.approx(2.0 * ref.cpu(), abs=1e-14, rel=1e-12) == out.cpu()


def test_dense_cell() -> None:
    """The evaluation of a cell over its images calls the damping."""
    structure = cell()
    c6 = _c6(structure)
    ref = dispersion_atm_periodic(structure, c6, param, cutoff=cutoff.disp3)
    out = dispersion_atm_periodic(
        structure, c6, param, damping=Doubled(), cutoff=cutoff.disp3
    )
    assert ref.abs().sum() > 0.0
    assert pytest.approx(2.0 * ref.cpu(), abs=1e-14, rel=1e-12) == out.cpu()


@pytest.mark.parametrize("kind", ["molecule", "cell"])
def test_sparse(kind: str) -> None:
    """The evaluation over a neighbour list calls the damping."""
    structure = molecule() if kind == "molecule" else cell()
    c6 = _c6(structure)
    nbl = build_neighborlist(structure, cutoff.disp3)
    ref = dispersion_atm_sparse(structure, c6, param, nbl, cutoff=cutoff.disp3)
    out = dispersion_atm_sparse(
        structure, c6, param, nbl, damping=Doubled(), cutoff=cutoff.disp3
    )
    assert ref.abs().sum() > 0.0
    assert pytest.approx(2.0 * ref.cpu(), abs=1e-14, rel=1e-12) == out.cpu()


@pytest.mark.parametrize("kind", ["molecule", "cell"])
def test_dftd3(kind: str) -> None:
    """A three-body damping given to `dftd3` reaches the ATM term."""
    structure = molecule() if kind == "molecule" else cell()
    two = RationalTwoBody()

    no_atm = dftd3(structure, param, cutoff=cutoff, damping=Damping(two))
    default = dftd3(structure, param, cutoff=cutoff)
    doubled = dftd3(
        structure, param, cutoff=cutoff, damping=Damping(two, Doubled())
    )
    e3 = default - no_atm
    assert e3.abs().sum() > 0.0
    assert pytest.approx(2.0 * e3.cpu(), abs=1e-14) == (doubled - no_atm).cpu()
