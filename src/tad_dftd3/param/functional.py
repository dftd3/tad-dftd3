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
Functional lookup
=================

Look up D3 damping parameters by functional name, resolving the same
name aliases as s-dftd3 (e.g. ``"b3lyp5"`` for ``"b3lyp"``, ``"pbeh"`` for
``"pbe0"``).

Example
-------
>>> import tad_dftd3 as d3
>>> param = d3.param.get_functional_params("b3lyp5", damping="bj")
>>> param["a1"]
tensor(0.3981)
"""

from __future__ import annotations

import os.path as op
from typing import Any

import torch
from tad_mctc.typing import DD, Tensor, get_default_device, get_default_dtype

try:
    import tomllib  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = ["get_functional_params"]

_META_KEYS = ("mbd", "damping", "doi")

_parameters: dict[str, Any] | None = None
_aliases: dict[str, str] | None = None


def _load_parameters() -> dict[str, Any]:
    global _parameters
    if _parameters is None:
        path = op.join(op.dirname(__file__), "parameters.toml")
        with open(path, "rb") as fh:
            _parameters = tomllib.load(fh)
    return _parameters


def _load_aliases() -> dict[str, str]:
    global _aliases
    if _aliases is None:
        path = op.join(op.dirname(__file__), "aliases.toml")
        with open(path, "rb") as fh:
            _aliases = dict(tomllib.load(fh)["alias"])
    assert _aliases is not None
    return _aliases


def _normalize(functional: str) -> str:
    """Normalize a functional name the same way s-dftd3's Fortran
    ``get_method_id`` normalizes its input: lowercase, strip hyphens."""
    return functional.lower().replace("-", "")


def _merge(
    entry: dict[str, Any],
    base: dict[str, Any],
    preference: list[str],
    keep_meta: bool,
) -> dict[str, Any]:
    for variant in preference:
        if variant not in entry:
            continue
        merged = {**base.get(variant, {}), **entry[variant]}
        if not keep_meta:
            for key in _META_KEYS:
                merged.pop(key, None)
        return merged

    raise ValueError(
        f"none of the requested damping variants {preference!r} are available "
        "for this functional"
    )


def get_functional_params(
    functional: str,
    damping: str | list[str] | None = None,
    keep_meta: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """
    Look up D3 damping parameters for a functional by name.

    Parameters
    ----------
    functional : str
        Name of the functional (case-insensitive, hyphen-insensitive).
        Recognizes the same synonyms as s-dftd3, e.g. ``"b3lyp5"`` and
        ``"pbeh"`` resolve to ``"b3lyp"`` and ``"pbe0"``, respectively.
    damping : str | list[str] | None, optional
        Damping variant(s) to look up (e.g. ``"bj"``, ``"zero"``, ``"bjm"``,
        ``"zerom"``, ``"op"``, ``"cso"``), tried in order. Defaults to the
        data base's own preference order (usually ``["bj", "zero"]``).
    keep_meta : bool, optional
        Keep the ``mbd``, ``damping`` and ``doi`` metadata fields in the
        returned dictionary instead of stripping them. Defaults to ``False``.
    dtype : torch.dtype | None, optional
        Floating point precision for the returned tensors.
    device : torch.device | None, optional
        Device for the returned tensors.

    Returns
    -------
    dict[str, Tensor]
        Damping parameters, ready to pass to :func:`tad_dftd3.dftd3`.

    Raises
    ------
    KeyError
        If ``functional`` (after alias resolution) is not in the data base.
    ValueError
        If none of the requested damping variants are available for this
        functional.
    """
    data = _load_parameters()
    aliases = _load_aliases()

    normalized = _normalize(functional)
    canonical = aliases.get(normalized, normalized)

    try:
        entry = data["parameter"][canonical]["d3"]
    except KeyError as e:
        raise KeyError(
            f"unknown functional {functional!r} (normalized: {canonical!r})"
        ) from e

    if damping is None:
        preference = data["default"]["d3"]
    elif isinstance(damping, str):
        preference = [damping]
    else:
        preference = damping

    base = data["default"]["parameter"]["d3"]
    merged = _merge(entry, base, preference, keep_meta)

    dd: DD = {
        "dtype": dtype if dtype is not None else get_default_dtype(),
        "device": device if device is not None else get_default_device(),
    }

    params: dict[str, Tensor] = {}
    for key, value in merged.items():
        if isinstance(value, (int, float)):
            params[key] = torch.tensor(float(value), **dd)
        else:
            params[key] = value

    return params
