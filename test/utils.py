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
Utility functions for testing.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

__all__ = ["approx_ref", "load_sample", "load_structure", "ref_tol"]


def load_structure(collection: str, record: str, dd: DD) -> Structure:
    """Look up `(collection, record)` via `get_structure`, moved to `dd`."""
    return get_structure(
        collection, record, device=dd["device"], dtype=dd["dtype"]
    )


def load_sample(collection: str, record: str, dd: DD) -> tuple[Tensor, Tensor]:
    """`load_structure`, keeping only `numbers`/`positions`."""
    structure = load_structure(collection, record, dd)
    return structure.numbers, structure.positions


_REF_TOL = {
    torch.float64: {"abs": 1e-12, "rel": 1e-12},
    torch.float32: {"abs": 1e-8, "rel": 1e-4},
}


def ref_tol(dtype: torch.dtype) -> dict[str, float]:
    """
    Absolute and relative tolerance for `pytest.approx` against the s-dftd3
    references, per dtype.

    Measured against s-dftd3 in float64, energies, gradients, Hessians, C6
    and weights agree to 1e-15 absolute or better (C6, up to 5e2, to 2e-16
    relative), so 1e-12 leaves three orders of magnitude of margin for other
    platforms and still catches any real change. In float32 the agreement is
    limited by the precision of the dtype (relative 2e-7 to 8e-6 measured).
    """
    return _REF_TOL[dtype]


def approx_ref(expected: Tensor, dtype: torch.dtype) -> object:
    """`pytest.approx` of `expected` at the :func:`ref_tol` of `dtype`."""
    tolerance = ref_tol(dtype)
    return pytest.approx(expected, abs=tolerance["abs"], rel=tolerance["rel"])
