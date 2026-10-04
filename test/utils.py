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

from tad_mctc.data.structures import get_structure
from tad_mctc.io.structure import Structure
from tad_mctc.typing import DD, Tensor

__all__ = ["load_sample", "load_structure"]


def load_structure(collection: str, record: str, dd: DD) -> Structure:
    """Look up `(collection, record)` via `get_structure`, moved to `dd`."""
    return get_structure(
        collection, record, device=dd["device"], dtype=dd["dtype"]
    )


def load_sample(collection: str, record: str, dd: DD) -> tuple[Tensor, Tensor]:
    """`load_structure`, keeping only `numbers`/`positions`."""
    structure = load_structure(collection, record, dd)
    return structure.numbers, structure.positions
