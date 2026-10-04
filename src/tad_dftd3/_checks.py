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
Argument checks shared by the public functions that take a
:class:`~tad_mctc.io.structure.Structure`.

They only look at Python types and at whether a field is set, never at
tensor values, so they are safe under ``vmap`` and ``torch.compile``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from tad_mctc.io.structure import Structure

__all__ = ["require_molecule", "takes_structure"]


F = TypeVar("F", bound=Callable[..., Any])


def takes_structure(func: F) -> F:
    """
    Reject a first argument that is not a ``Structure``.

    Up to 0.7.0, the public functions took ``numbers`` and ``positions`` as
    their first two arguments. Such a call would otherwise fail with an
    unhelpful message about the number of positional arguments, or not at
    all for functions whose next argument also takes a tensor.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        structure = args[0] if args else kwargs.get("structure")
        if not isinstance(structure, Structure):
            raise TypeError(
                f"'{func.__name__}' takes a 'tad_mctc.Structure' as its "
                f"first argument, not '{type(structure).__name__}'. Up to "
                "0.7.0 it took 'numbers' and 'positions'; pass "
                "'Structure(numbers=numbers, positions=positions)' instead."
            )
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def require_molecule(structure: Structure, term: str) -> None:
    """
    Raise if `structure` is a periodic cell, for a `term` that has no
    periodic evaluation. Evaluating the molecular one instead would
    silently ignore the periodic images.
    """
    if structure.lattice is not None:
        raise ValueError(
            f"The {term} has no periodic evaluation, but 'structure' has a "
            "'lattice'."
        )
