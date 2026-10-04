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
"""
Checks shared by the sparse molecular and periodic paths: input errors.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.neighbor.images import build_periodic_shifts
from tad_mctc.neighbor.list import build_neighborlists
from tad_mctc.typing import DD

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import cells, param_on
from ..conftest import DEVICE

cutoff = Cutoff(cn=15.0, disp2=20.0)

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def test_shifts_and_list_exclusive() -> None:
    structure = cells["urea"]
    p = param_on(DD64)
    shifts = build_periodic_shifts(
        structure.lattice, structure.periodic, cutoff.disp2
    )
    (nbl,) = build_neighborlists(structure, (cutoff.disp2,))

    with pytest.raises(ValueError, match="not both"):
        dftd3(structure, p, cutoff=cutoff, shifts=shifts, nbl_disp2=nbl)

    with pytest.raises(ValueError, match="replace"):
        dftd3(structure, p, cutoff=cutoff, shifts=shifts, sparse=True)


def test_list_too_short_raises() -> None:
    structure = cells["urea"]
    p = param_on(DD64)

    (small,) = build_neighborlists(structure, (cutoff.disp2 - 5.0,))
    with pytest.raises(ValueError):
        dftd3(structure, p, cutoff=cutoff, nbl_disp2=small)
