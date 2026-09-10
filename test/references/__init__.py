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
Coordination number, reference-system weights and atomic C6 coefficients
from the s-dftd3 Fortran *library*, for every molecule any test in this
suite compares against it. None of these three quantities are reachable
through the ``dftd3`` Python package (see
``tools/refs/gen_refs_fortran.f90`` for why), so unlike the
energy/gradient/Hessian references in ``test/reference.py``, they are not
computed live at test time -- one JSON file per molecule is dumped once,
by ``tools/refs``, and committed here instead.

All three were generated at the coordination-number cutoff ``dftd3()``
uses internally (``tad_dftd3.defaults.D3_CN_CUTOFF``), so ``reference_c6``
is usable wherever an independently computed C6 is needed -- including as a
drop-in for ``disp.dispersion()`` / ``damping.dispersion_atm()`` in
``test_disp``, without going through tad-dftd3's own code first.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from tad_mctc.typing import DD, Tensor

__all__ = ["reference_cn", "reference_weights", "reference_c6"]

_DATA_DIR = Path(__file__).parent


@lru_cache
def _data(name: str) -> dict:
    return json.loads((_DATA_DIR / f"{name}.json").read_text())


def reference_cn(name: str, dd: DD) -> Tensor:
    """Coordination number for molecule ``name``, shape ``(nat,)``."""
    return torch.tensor(_data(name)["cn"], dtype=torch.double).to(**dd)


def reference_weights(name: str, dd: DD) -> Tensor:
    """
    Reference-system weights for molecule ``name``, shape ``(nat, 7)``
    -- zero-padded out to tad-dftd3's fixed width, see
    ``tad_dftd3.reference.Reference``.
    """
    return torch.tensor(_data(name)["weights"], dtype=torch.double).to(**dd)


def reference_c6(name: str, dd: DD) -> Tensor:
    """Atomic C6 coefficients for molecule ``name``, shape ``(nat, nat)``."""
    return torch.tensor(_data(name)["c6"], dtype=torch.double).to(**dd)
