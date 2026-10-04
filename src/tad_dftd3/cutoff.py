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
r"""
Real-space cutoffs
==================

One real-space cutoff per part of the D3 model, since they decay at very
different rates: the counting function behind the coordination number
exponentially, the two-body energy as :math:`R^{-6}`, the three-body term
as :math:`R^{-9}`.

Mirrors ``realspace_cutoff`` in s-dftd3's ``dftd3_cutoff``, defaults
included, so both implementations discard the same pairs and triples.
Like s-dftd3, the two dispersion terms can be switched off smoothly instead
of abruptly: with a width :math:`w > 0` (``width2``, ``width3``), the
contribution of a distance :math:`r` between :math:`R_\text{cut} - w` and
:math:`R_\text{cut}` is scaled by the quintic

.. math::

    s(r) = x^3 \left(10 - 15 x + 6 x^2\right), \qquad
    x = \dfrac{R_\text{cut} - r}{w},

which is 1 at the inner edge and 0 at the cutoff, with vanishing first and
second derivatives at both. The two-body term is scaled by :math:`s(r_{ij})`,
the three-body term by :math:`s(r_{ij}) s(r_{ik}) s(r_{jk})`. The widths
default to zero, the hard cutoff, as in s-dftd3. There is no width for the
coordination number.

Example
-------
>>> from tad_dftd3.cutoff import Cutoff
>>> cutoff = Cutoff()
>>> print(cutoff)
Cutoff(cn=40.0, disp2=60.0, disp3=40.0, width2=0.0, width3=0.0)
>>> cutoff = Cutoff(disp3=25.0, width2=5.0)
>>> print(cutoff.disp3, cutoff.width2)
25.0 5.0
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real

import torch
from tad_mctc.typing import Tensor

from . import defaults

__all__ = ["Cutoff", "smooth_cutoff"]


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

    width2: float = defaults.D3_DISP2_WIDTH
    """
    Width of the smooth two-body cutoff. Zero (default) is a hard cutoff;
    a width of at least `disp2` is treated as zero, as in s-dftd3.
    """

    width3: float = defaults.D3_DISP3_WIDTH
    """
    Width of the smooth three-body cutoff, like `width2` for `disp3`.
    """

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

        for name in ("width2", "width3"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"Cutoff '{name}' must not be negative.")


def smooth_cutoff(distance: Tensor, cutoff: float, width: float) -> Tensor:
    r"""
    Smooth switch-off of a contribution at the real-space `cutoff`, as in
    s-dftd3's ``smooth_cutoff``: 1 up to ``cutoff - width``, 0 from `cutoff`,
    and the quintic :math:`x^3 (10 - 15 x + 6 x^2)` of
    :math:`x = (\text{cutoff} - r) / \text{width}` in between.

    A `width` of zero, or of at least `cutoff`, is a hard cutoff, where the
    switch is 1 everywhere; the pairs beyond the cutoff are dropped by the
    caller's mask. `cutoff` and `width` are plain floats, so the choice is a
    constant to :func:`torch.compile` and :func:`torch.vmap`.

    Differentiable to any order for ``distance > 0``, with finite derivatives
    also at both edges, where it is :math:`C^2`.

    Parameters
    ----------
    distance : Tensor
        Distances, in Bohr.
    cutoff : float
        Real-space cutoff, in Bohr.
    width : float
        Width of the switching region below `cutoff`, in Bohr.

    Returns
    -------
    Tensor
        The switch, of the shape of `distance`.
    """
    if width <= 0.0 or width >= cutoff:
        return torch.ones_like(distance)

    # Clamped, the polynomial is exactly the piecewise function: 1 for
    # `x >= 1` (inside the switching region) and 0 for `x <= 0`.
    x = torch.clamp((cutoff - distance) / width, min=0.0, max=1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))
