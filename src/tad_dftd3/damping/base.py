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
Damping selection
=================

The two- and the three-body damping are chosen independently, like
``damping_type`` of ``dftd`` (e.g. rational damping of the two-body term and
zero damping of the three-body term), and both read one shared
:class:`~tad_dftd3.damping.param.DampingParam`. A damping is a static,
hashable value without tensors; it declares which parameters it needs and
which it defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

from tad_mctc.typing import Tensor

from .param import DampingParam

__all__ = [
    "DFTD_RADIUS_DEFAULTS",
    "Damping",
    "PairData",
    "ThreeBodyDamping",
    "TripleData",
    "TwoBodyDamping",
    "pow6",
    "pow6_pow8",
    "scaled_radius",
]


class PairData(NamedTuple):
    """
    What the two-body damping knows about the pairs it damps. Each entry is
    broadcast against `distances`: the shape of `distances` for a molecule,
    and with one more axis of the images for a cell, where the others have
    a trailing axis of length one.
    """

    distances: Tensor
    """Pairwise distances."""

    qq: Tensor
    """Quotient of C8 and C6 dispersion coefficients,
    ``3 * r4r2[i] * r4r2[j]``."""

    c6: Tensor
    """C6 dispersion coefficients."""

    rvdw: Tensor
    """Van-der-Waals radii of the pairs. One where a padding atom is
    involved."""

    znum: Tensor
    """Sum of the atomic numbers of the pairs. One where only padding atoms
    are involved."""

    rdamp: Tensor
    """Damping radius of the pairs that the model supplies, like dftd's
    ``get_2b_rdamp``: ``sqrt(qq)`` in D3, as in D4. Read by the dampings
    ported from dftd, which scale it to ``a1 * rdamp + a2``."""


class TripleData(NamedTuple):
    """
    What the three-body damping knows about the triples it damps, like the
    arguments of dftd's ``get_3b_damp``. The entries broadcast against each
    other but need not have the full shape of the triples (the dense
    evaluation of a molecule passes ``(..., nat, nat, 1)``-like views), so a
    damping must not assume a shape.

    Every entry is finite and positive, also for triples that are dropped
    and for padding atoms, which are masked by the caller afterwards: a
    damping needs no masking of its own.

    D3 supplies two radii per pair, and each damping reads the one it is
    defined with: the van-der-Waals radii (`rvdw*`) for the zero damping of
    s-dftd3, the damping radii of the model (`rdamp*`, ``sqrt(qq)`` as in D4)
    for the dampings ported from dftd.
    """

    r: Tensor
    """Product of the three distances of a triple."""

    r2ij: Tensor
    """Squared distance between atoms i and j."""

    r2ik: Tensor
    """Squared distance between atoms i and k."""

    r2jk: Tensor
    """Squared distance between atoms j and k."""

    rdampij: Tensor
    """Damping radius of the pair i, j that the model supplies, like dftd's
    ``get_3b_rdamp``."""

    rdampik: Tensor
    """Damping radius of the pair i, k."""

    rdampjk: Tensor
    """Damping radius of the pair j, k."""

    rvdwij: Tensor
    """Van-der-Waals radius of the pair i, j."""

    rvdwik: Tensor
    """Van-der-Waals radius of the pair i, k."""

    rvdwjk: Tensor
    """Van-der-Waals radius of the pair j, k."""


class _Damping:
    """The parameters shared by the two- and three-body damping."""

    defaults: ClassVar[tuple[tuple[str, float], ...]] = ()
    """Values for parameters that are not set. A tuple, not a dictionary:
    ``torch.compile`` cannot trace the methods of a class-level dict."""

    label: ClassVar[str] = ""

    def value(self, param: DampingParam, name: str) -> Tensor | float:
        """
        The parameter `name`, or the default of this damping if it is not
        set.

        Raises
        ------
        ValueError
            If the parameter is not set and this damping has no default.
        """
        value = getattr(param, name)
        if value is not None:
            return value

        for key, default in self.defaults:
            if key == name:
                return default

        raise ValueError(
            f"{self.label} requires the damping parameter '{name}', which is "
            "not set."
        )


DFTD_RADIUS_DEFAULTS: tuple[tuple[str, float], ...] = (("a1", 1.0), ("a2", 0.0))
"""dftd's defaults of `a1` and `a2`: the damping radius as it is."""


def pow6(x: Tensor) -> Tensor:
    """
    ``x**6`` by multiplication: in eager mode, ``torch.pow`` with an integer
    exponent is several times slower, which shows in the per-pair kernels.
    """
    x2 = x * x
    return x2 * x2 * x2


def pow6_pow8(x: Tensor) -> tuple[Tensor, Tensor]:
    """``x**6`` and ``x**8`` by multiplication, see :func:`pow6`."""
    x2 = x * x
    x6 = x2 * x2 * x2
    return x6, x6 * x2


def scaled_radius(
    damping: _Damping, param: DampingParam, rdamp: Tensor
) -> Tensor:
    """
    The critical radius ``a1 * rdamp + a2``, the first-order polynomial
    scaling of a damping radius, of the rational damping and those built on
    it. `a1` and `a2` are required unless `damping` defaults them (as the
    dampings ported from dftd do, to 1 and 0).
    """
    return damping.value(param, "a1") * rdamp + damping.value(param, "a2")


class TwoBodyDamping(_Damping):
    """
    Damping of the two-body term. A subclass implements ``__call__``,
    reading each parameter with :meth:`value`, and declares its
    `defaults`.
    """

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        """
        Damped dispersion of the pairs divided by their C6, i.e. the
        energy of a pair is ``-c6`` times this (half of it for each atom)
        before the cutoff is applied.

        With the scalings of the C6 and the C8 term that is
        ``s6 * f6 / r**6 + s8 * qq * f8 / r**8``, with the damping functions
        `f6` and `f8` of the variant.

        Parameters
        ----------
        pairs : PairData
            The pairs.
        param : DampingParam
            Parameters.

        Returns
        -------
        Tensor
            Value for each pair.
        """
        raise NotImplementedError


class ThreeBodyDamping(_Damping):
    """
    Damping of the three-body term. A subclass implements ``__call__``,
    reading each parameter with :meth:`value`, and declares its `defaults`.
    The geometry of the term is evaluated by the functions of
    :mod:`tad_dftd3.damping.atm` and :mod:`tad_dftd3.sparse`, which call the
    damping for every triple.
    """

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        """
        Damping factor of the triples, with the scaling `s9` of the term,
        like ``d9`` of dftd: the energy of a triple is
        ``-sqrt(|c6_ij c6_ik c6_jk|)`` times this and its angular factor,
        a third of it for each atom, before the cutoff is applied.

        Parameters
        ----------
        triples : TripleData
            The triples.
        param : DampingParam
            Parameters.

        Returns
        -------
        Tensor
            Value for each triple.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Damping:
    """
    A combination of two- and three-body damping.

    Parameters
    ----------
    two : TwoBodyDamping
        Damping of the two-body term.
    three : ThreeBodyDamping | None, optional
        Damping of the three-body term. ``None`` (default) is no three-body
        term.
    """

    two: TwoBodyDamping
    three: ThreeBodyDamping | None = None
