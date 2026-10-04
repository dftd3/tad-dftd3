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
from the s-dftd3 Fortran *library*, for every structure a test compares
against it. None of these three quantities are reachable through the
``dftd3`` Python package (see ``tools/refs/gen_refs_fortran.f90`` for why),
so unlike the energy/gradient/Hessian references in ``test/reference.py``,
they are not computed live at test time: one JSON file per structure is
dumped once by ``tools/refs`` and committed here, as
``<collection>/<record>.json`` for the ``(collection, record)`` of
:func:`tad_mctc.data.structures.get_structure`.

All three were generated at the coordination-number cutoff ``dftd3()``
uses internally (``tad_dftd3.defaults.D3_CN_CUTOFF``), so ``reference_c6``
is usable wherever an independently computed C6 is needed, without going
through tad-dftd3's own code first.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from tad_mctc.typing import DD, Tensor

__all__ = ["reference_cn", "reference_weights", "reference_c6"]

_DATA_DIR = Path(__file__).parent


@lru_cache
def _data(collection: str, record: str) -> dict[str, Any]:
    path = _DATA_DIR / collection / f"{record}.json"
    return json.loads(path.read_text())


def reference_cn(collection: str, record: str, dd: DD) -> Tensor:
    """Coordination number of a structure, shape ``(nat,)``."""
    data = _data(collection, record)
    return torch.tensor(data["cn"], dtype=torch.double).to(**dd)


def reference_weights(collection: str, record: str, dd: DD) -> Tensor:
    """
    Reference-system weights of a structure, shape ``(nat, 7)``, zero-padded
    out to tad-dftd3's fixed width, see ``tad_dftd3.reference.Reference``.
    """
    data = _data(collection, record)
    return torch.tensor(data["weights"], dtype=torch.double).to(**dd)


def reference_c6(collection: str, record: str, dd: DD) -> Tensor:
    """Atomic C6 coefficients of a structure, shape ``(nat, nat)``."""
    data = _data(collection, record)
    return torch.tensor(data["c6"], dtype=torch.double).to(**dd)
