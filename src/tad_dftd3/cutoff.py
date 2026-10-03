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
Real-space cutoffs
==================

One real-space cutoff per part of the D3 model, since they decay at very
different rates: the counting function behind the coordination number
exponentially, the two-body energy as :math:`R^{-6}`, the three-body term
as :math:`R^{-9}`.

Mirrors ``realspace_cutoff`` in s-dftd3's ``dftd3_cutoff``, defaults
included, so both implementations discard the same pairs and triples.
s-dftd3 can also switch the dispersion terms off smoothly over a width
(``width2``/``width3``); those default to zero there, i.e. the hard cutoff
implemented here.

Example
-------
>>> from tad_dftd3.cutoff import Cutoff
>>> cutoff = Cutoff()
>>> print(cutoff)
Cutoff(cn=40.0, disp2=60.0, disp3=40.0)
>>> cutoff = Cutoff(disp3=25.0)
>>> print(cutoff.disp3)
25.0
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

from . import defaults

__all__ = ["Cutoff"]


@dataclass(frozen=True)
class Cutoff:
    """
    Real-space cutoffs for the individual parts of the D3 model, in Bohr.

    Plain floats, like the ``cutoff`` of the ``tad-mctc`` coordination number
    models: they only mask distances, so they are constants to
    :func:`torch.compile`, :func:`torch.vmap` and :func:`torch.func.jacrev`
    and need no device or dtype. Any real number, including NumPy scalars,
    is converted to :class:`float`; tensors are rejected.
    """

    cn: float = defaults.D3_CN_CUTOFF
    """Coordination number cutoff."""

    disp2: float = defaults.D3_DISP2_CUTOFF
    """Two-body dispersion interaction cutoff."""

    disp3: float = defaults.D3_DISP3_CUTOFF
    """Three-body dispersion interaction cutoff."""

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"Cutoff '{name}' must be a plain number, not "
                    f"'{type(value).__name__}'."
                )
            object.__setattr__(self, name, float(value))
