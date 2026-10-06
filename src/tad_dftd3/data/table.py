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
Data: Element tables
====================

Resolution of the per-element tables (indexed by atomic number) that the
element parameters `rcov_table`, `rvdw_table` and `r4r2_table` are given as.
"""

from __future__ import annotations

from tad_mctc.data import radii, resolve_table
from tad_mctc.typing import TableFunction, Tensor

from .r4r2 import R4R2

__all__ = ["TABLES", "element_table"]


TABLES: dict[str, tuple[TableFunction, tuple[int, ...]]] = {
    "rcov_table": (radii.COV_D3, (119,)),
    "r4r2_table": (R4R2, (119,)),
    "rvdw_table": (radii.VDW_PAIRWISE, (104, 104)),
}
"""Default and required shape of each per-element table, by argument name."""


def element_table(
    table: Tensor | TableFunction | None, name: str, like: Tensor
) -> Tensor:
    """
    Resolve a per-element table on the device and dtype of `like`.

    Tables built from a :class:`~tad_mctc.typing.TableFunction` (including
    the default) are cached per device and dtype by
    :func:`tad_mctc.data.resolve_table`, for as long as the function exists.

    A table that is passed must have the shape given in :data:`TABLES` for
    `name`, e.g. ``(119,)`` for ``r4r2_table``. Only the shape is checked,
    which does not break the graph under ``torch.compile``.

    Parameters
    ----------
    table : Tensor | TableFunction | None
        The table, or ``None`` for the default of `name`.
    name : str
        Name of the argument, a key of :data:`TABLES`.
    like : Tensor
        Tensor whose device and dtype the table is resolved to.

    Returns
    -------
    Tensor
        The table. Callers must treat it as read-only, since it may be
        shared through the cache.

    Raises
    ------
    ValueError
        If the table does not have the required shape.
    """
    default, shape = TABLES[name]
    if table is None:
        return resolve_table(default, like)

    out = resolve_table(table, like)
    if out.shape != shape:
        raise ValueError(
            f"'{name}' is a table indexed by atomic number, of shape "
            f"{shape}, not one value per atom or atom pair (got shape "
            f"{tuple(out.shape)})."
        )

    return out
