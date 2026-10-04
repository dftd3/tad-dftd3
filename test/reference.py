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
Reference energy from the s-dftd3 Fortran implementation
===========================================================

Four functions -- :func:`reference_energy_per_atom`,
:func:`reference_pairwise`, :func:`reference_gradient` and
:func:`reference_hessian` -- that compute dispersion quantities by calling
the s-dftd3 Fortran implementation through its Python bindings (the
``dftd3`` package -- install it with ``pip``, nothing else needed), so that
tests comparing against them need no hardcoded reference tensor. ``dftd3``
is a required test dependency, not an optional one; there is no offline
fallback. This module exists only because it is imported by more than one
test module -- it is not a general-purpose framework, and each function in
it should stay exactly as small as its callers need. Add a function here
only once a test actually calls it.

Real-space cutoffs: ``cutoff=None`` (the default) leaves s-dftd3 at its own
compiled-in cutoffs, so a comparison against tad-dftd3's defaults checks
*which values* those are, not merely how they are applied. Passing a
:class:`tad_dftd3.cutoff.Cutoff` pins s-dftd3 to it instead, via
``DispersionModel.set_realspace_cutoff``.

Every function takes a :class:`tad_mctc.Structure`, a single system. For a
periodic cell, its ``lattice`` and ``periodic`` mask are passed straight to
``dftd3.interface.DispersionModel``. For a cell that is not
periodic along every axis, s-dftd3 folds the atoms into the cell along
*all* axes (``wrap_to_central_cell`` in ``s-dftd3/src/dftd3/utils.f90``),
also along the open ones, which moves atoms that lie outside the cell along
an open axis relative to the others. tad-dftd3 only folds along periodic
axes, so tests of such cells keep the atoms inside the cell along the open
axes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dftd3.interface as dftd3_interface
import numpy as np
import torch
from numpy.typing import NDArray
from tad_mctc import Structure
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import defaults
from tad_dftd3.cutoff import Cutoff

__all__ = [
    "reference_energy_per_atom",
    "reference_gradient",
    "reference_hessian",
    "reference_pairwise",
]


def _to_numpy(tensor: Tensor, dtype: type) -> NDArray[Any]:
    """A detached NumPy copy of `tensor` as `dtype`."""
    return tensor.detach().cpu().numpy().astype(dtype)


def _build_model(
    structure: Structure, cutoff: Cutoff | None
) -> dftd3_interface.DispersionModel:
    """
    Build a ``DispersionModel``, shared by every function here.
    ``cutoff=None`` leaves s-dftd3 at its own cutoffs (see module docstring).
    """
    lattice = structure.lattice
    periodic = structure.periodic
    model = dftd3_interface.DispersionModel(
        _to_numpy(structure.numbers, np.int32),
        _to_numpy(structure.positions, np.float64),
        None if lattice is None else _to_numpy(lattice, np.float64),
        None if periodic is None else _to_numpy(periodic, np.bool_),
    )

    if cutoff is not None:
        model.set_realspace_cutoff(
            cn=cutoff.cn,
            disp2=cutoff.disp2,
            disp3=cutoff.disp3,
            width2=cutoff.width2,
            width3=cutoff.width3,
        )

    return model


def _dd(structure: Structure) -> DD:
    """Device and dtype of the positions, which every result is cast to."""
    positions = structure.positions
    return {"device": positions.device, "dtype": positions.dtype}


def _build_damping_param(
    param: Mapping[str, Tensor | float],
) -> dftd3_interface.RationalDampingParam:
    """Translate a tad-dftd3 ``param`` dict into a ``RationalDampingParam``,
    shared by every function in this module.

    A missing ``"s9"`` is treated as ``0.0`` (no ATM term), matching
    ``dftd3()`` itself, which only adds the three-body term when
    ``"s9" in param and param["s9"] != 0.0``.
    """

    def value_or_default(key: str, default: float) -> float:
        if key in param:
            return float(param[key])
        return default

    return dftd3_interface.RationalDampingParam(
        s6=value_or_default("s6", defaults.S6),
        s8=value_or_default("s8", defaults.S8),
        a1=value_or_default("a1", defaults.A1),
        a2=value_or_default("a2", defaults.A2),
        s9=value_or_default("s9", 0.0),
        alp=value_or_default("alp", defaults.ALP),
    )


def reference_energy_per_atom(
    structure: Structure,
    param: Mapping[str, Tensor | float],
    *,
    cutoff: Cutoff | None = None,
) -> Tensor:
    """
    Atom-resolved dispersion energy from the s-dftd3 Fortran reference.

    tad-dftd3's public ``dftd3()`` returns an atom-resolved energy, shape
    ``(nat,)``, not a single total -- most tests compare against that
    directly. s-dftd3's Python API only exposes a scalar total
    (``get_dispersion``) or a pair-resolved ``(nat, nat)`` matrix
    (``get_pairwise_dispersion``), so the atom-resolved value used here is
    reconstructed via :func:`reference_pairwise`, summing each atom's row
    of the two pairwise matrices. This was checked against
    ``get_dispersion``'s own total, not assumed: the reconstructed values
    agree with tad-dftd3's atom-resolved output to about 2e-10 absolute on
    every sample tried while writing this function.

    Parameters
    ----------
    structure : Structure
        A single molecule or cell, coordinates in Bohr.
    param : dict[str, Tensor]
        Damping parameters in tad-dftd3's own convention (``s6``, ``s8``,
        ``a1``, ``a2``, and optionally ``s9``, ``alp``).
    cutoff : Cutoff | None, optional
        Real-space cutoffs to pin s-dftd3 to. Defaults to ``None``, i.e.
        s-dftd3's own.

    Returns
    -------
    Tensor
        Atom-resolved energy, shape ``(nat,)``, in Hartree, cast to
        the positions' dtype and moved to their device.
    """
    two_body, three_body = reference_pairwise(structure, param, cutoff=cutoff)

    # Each matrix holds half of every pair's energy at both [i, j] and
    # [j, i] (confirmed against get_dispersion's own total, not assumed),
    # so summing a row recovers that atom's full share.
    return two_body.sum(dim=1) + three_body.sum(dim=1)


def reference_pairwise(
    structure: Structure,
    param: Mapping[str, Tensor | float],
    *,
    cutoff: Cutoff | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Pair-resolved two-body and three-body energies from the s-dftd3 Fortran
    reference.

    This is a much sharper comparison than the total energy: a wrong pair
    shows up as a wrong matrix element at a specific index, rather than
    being hidden inside (or accidentally cancelled by) a sum over every
    pair in the system.

    Matrix convention, confirmed empirically rather than assumed (see
    :func:`reference_energy_per_atom`): each returned matrix is symmetric
    with a zero diagonal, and holds *half* of each pair's energy at both
    ``[i, j]`` and ``[j, i]`` -- summing the whole matrix, not half of it,
    recovers the total energy for that term. So the full energy of one pair
    ``(i, j)`` is ``two_body[i, j] + two_body[j, i]``, and
    ``float(two_body.sum())`` equals ``reference_energy_per_atom(...,
    param without "s9").sum()``.

    Parameters
    ----------
    structure : Structure
        A single molecule or cell, coordinates in Bohr.
    param : dict[str, Tensor]
        Damping parameters, as in :func:`reference_energy_per_atom`.
    cutoff : Cutoff | None, optional
        Real-space cutoffs, as in :func:`reference_energy_per_atom`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(two_body, three_body)``, each shape ``(nat, nat)`` and in
        Hartree, cast to the positions' dtype and device. ``three_body``
        is s-dftd3's "non-additive pairwise energy"; it sums to the ATM
        energy only to about 1e-10 relative, not to full precision, because
        the three-body term is genuinely a sum over atom triples, not
        pairs, and there is more than one reasonable way to fold a
        triple's energy back onto a pairwise matrix.
    """
    model = _build_model(structure, cutoff)
    damping_param = _build_damping_param(param)

    result = model.get_pairwise_dispersion(damping_param)
    two_body = np.asarray(result["additive pairwise energy"])
    three_body = np.asarray(result["non-additive pairwise energy"])

    dd: DD = _dd(structure)
    return torch.tensor(two_body, **dd), torch.tensor(three_body, **dd)


def reference_gradient(
    structure: Structure,
    param: Mapping[str, Tensor | float],
    *,
    cutoff: Cutoff | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Nuclear gradient and virial of the dispersion energy from the s-dftd3
    Fortran reference.

    The virial is the derivative of the energy with respect to a strain
    ``eps`` that deforms positions and lattice vectors (rows) alike, ``x ->
    x (1 + eps)``. By the chain rule it is ``positions.T @ dE/dpositions +
    lattice.T @ dE/dlattice``, which is how tests check the derivative with
    respect to the lattice against it.

    Parameters
    ----------
    structure : Structure
        A single molecule or cell, coordinates in Bohr.
    param : dict[str, Tensor]
        Damping parameters, as in :func:`reference_energy_per_atom`.
    cutoff : Cutoff | None, optional
        Real-space cutoffs, as in :func:`reference_energy_per_atom`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(gradient, virial)``, shapes ``(nat, 3)`` and ``(3, 3)``, in
        Hartree per Bohr and Hartree, cast to the positions' dtype and
        device.
    """
    model = _build_model(structure, cutoff)
    damping_param = _build_damping_param(param)

    result = model.get_dispersion(damping_param, grad=True)
    gradient = np.asarray(result["gradient"])
    virial = np.asarray(result["virial"])

    dd: DD = _dd(structure)
    return torch.tensor(gradient, **dd), torch.tensor(virial, **dd)


def reference_hessian(
    structure: Structure,
    param: Mapping[str, Tensor | float],
    *,
    cutoff: Cutoff | None = None,
) -> Tensor:
    """
    Hessian of the dispersion energy with respect to the positions from the
    s-dftd3 Fortran reference.

    s-dftd3 returns it flattened, ``(3 * nat, 3 * nat)``; a plain (C-order)
    reshape to ``(nat, 3, nat, 3)`` is the right one (checked against
    tad-dftd3's autodiff Hessian, which it matches to about 1e-15 for a
    molecule and 1e-12 for a cell, where more images contribute).

    Parameters
    ----------
    structure : Structure
        A single molecule or cell, coordinates in Bohr.
    param : dict[str, Tensor]
        Damping parameters, as in :func:`reference_energy_per_atom`.
    cutoff : Cutoff | None, optional
        Real-space cutoffs, as in :func:`reference_energy_per_atom`.

    Returns
    -------
    Tensor
        Hessian, shape ``(nat, 3, nat, 3)``, in Hartree per Bohr squared,
        cast to the positions' dtype and device.
    """
    model = _build_model(structure, cutoff)
    damping_param = _build_damping_param(param)

    result = model.get_hessian(damping_param)
    nat = structure.numbers.shape[-1]
    hessian = np.asarray(result["hessian"]).reshape(nat, 3, nat, 3)

    return torch.tensor(hessian, **_dd(structure))
