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

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from tad_mctc.data import radii, resolve_table
from tad_mctc.typing import TableFunction, Tensor

from .r4r2 import R4R2

__all__ = ["TABLES", "element_table", "reject_renamed_tables"]


TABLES: dict[str, tuple[TableFunction, tuple[int, ...]]] = {
    "rcov_table": (radii.COV_D3, (119,)),
    "r4r2_table": (R4R2, (119,)),
    "rvdw_table": (radii.VDW_PAIRWISE, (104, 104)),
}
"""Default and required shape of each per-element table, by argument name."""


_RENAMED = {name.removesuffix("_table"): name for name in TABLES}
"""Names up to 0.7.0 (per-atom values) mapped to their successor (tables)."""

F = TypeVar("F", bound=Callable[..., Any])


def reject_renamed_tables(func: F) -> F:
    """
    Reject the per-atom element parameters of 0.7.0 and earlier by name.

    Up to 0.7.0, ``rcov``, ``rvdw`` and ``r4r2`` took per-atom (or per-pair)
    values. They are now tables passed as ``rcov_table``, ``rvdw_table`` and
    ``r4r2_table``. The old names are rejected outright rather than checked
    by shape, since per-atom values of a single system with as many atoms as
    the table has entries would pass any shape check.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for old, new in _RENAMED.items():
            if old in kwargs:
                raise TypeError(
                    f"'{func.__name__}' no longer takes '{old}'. Up to 0.7.0 "
                    f"it took per-atom values, i.e. 'table[numbers]'; pass the "
                    f"per-element table itself as '{new}' instead."
                )
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


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
