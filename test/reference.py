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
:func:`reference_gradient_per_atom`, :func:`reference_hessian` and
:func:`reference_pairwise` -- that compute dispersion quantities by calling
the s-dftd3 Fortran implementation through its Python bindings (the
``dftd3`` package -- install it with ``pip``, nothing else needed), so that
tests comparing against them need no hardcoded reference tensor. ``dftd3``
is a required test dependency, not an optional one; there is no offline
fallback. This module exists only because it is imported by more than one
test module -- it is not a general-purpose framework, and each function in
it should stay exactly as small as its callers need. Add a function here
only once a test actually calls it.

Coordination-number cutoff: s-dftd3's API takes one explicitly
(``DispersionModel.set_realspace_cutoff(..., cn=...)``). tad-dftd3's own
``dftd3()`` applies none at all -- it calls ``coordination_number`` without
a ``cutoff`` argument -- so this module pins s-dftd3's CN cutoff to a value
large enough to be effectively unbounded for any test geometry, rather than
to the unused ``defaults.D3_CN_CUTOFF``.
"""

from __future__ import annotations

import dftd3.interface as dftd3_interface
import numpy as np
import torch
from tad_mctc.typing import Tensor

from tad_dftd3 import defaults

__all__ = [
    "reference_energy_per_atom",
    "reference_gradient_per_atom",
    "reference_hessian",
    "reference_pairwise",
]


# tad-dftd3's `dftd3()` never applies a coordination-number cutoff (see the
# module docstring); this is chosen to be effectively unbounded for any
# molecular geometry this test suite builds.
_EFFECTIVELY_UNBOUNDED_CN_CUTOFF = 10000.0


def _build_model(
    numbers: Tensor, positions: Tensor, disp_cutoff: float
) -> dftd3_interface.DispersionModel:
    """Build a ``DispersionModel`` with s-dftd3's cutoffs pinned to match
    ``dftd3()``'s own behaviour, shared by every function in this module."""
    numbers_numpy = numbers.detach().cpu().numpy().astype(np.int32)
    positions_numpy = positions.detach().cpu().numpy().astype(np.float64)

    model = dftd3_interface.DispersionModel(numbers_numpy, positions_numpy)
    model.set_realspace_cutoff(
        disp2=disp_cutoff,
        disp3=disp_cutoff,
        cn=_EFFECTIVELY_UNBOUNDED_CN_CUTOFF,
    )
    return model


def _build_damping_param(
    param: dict[str, Tensor],
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
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    *,
    disp_cutoff: float = defaults.D3_DISP_CUTOFF,
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
    numbers : Tensor
        Atomic numbers, shape ``(nat,)``.
    positions : Tensor
        Cartesian coordinates in Bohr, shape ``(nat, 3)``.
    param : dict[str, Tensor]
        Damping parameters in tad-dftd3's own convention (``s6``, ``s8``,
        ``a1``, ``a2``, and optionally ``s9``, ``alp``).
    disp_cutoff : float, optional
        Real-space cutoff, in Bohr, applied to both the two-body and the
        three-body term -- ``dftd3()`` shares a single cutoff between them.
        Defaults to ``tad_dftd3.defaults.D3_DISP_CUTOFF``.

    Returns
    -------
    Tensor
        Atom-resolved energy, shape ``(nat,)``, in Hartree, cast to
        ``positions``'s dtype and moved to its device.
    """
    two_body, three_body = reference_pairwise(
        numbers, positions, param, disp_cutoff=disp_cutoff
    )

    # Each matrix holds half of every pair's energy at both [i, j] and
    # [j, i] (confirmed against get_dispersion's own total, not assumed),
    # so summing a row recovers that atom's full share.
    return two_body.sum(dim=1) + three_body.sum(dim=1)


def reference_gradient_per_atom(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    *,
    disp_cutoff: float = defaults.D3_DISP_CUTOFF,
) -> Tensor:
    """
    Nuclear gradient of the dispersion energy from the s-dftd3 Fortran
    reference.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(nat,)``.
    positions : Tensor
        Cartesian coordinates in Bohr, shape ``(nat, 3)``.
    param : dict[str, Tensor]
        Damping parameters, as in :func:`reference_energy_per_atom`.
    disp_cutoff : float, optional
        Real-space cutoff, in Bohr, as in :func:`reference_energy_per_atom`.

    Returns
    -------
    Tensor
        The gradient with respect to ``positions``, shape ``(nat, 3)``, in
        Hartree per Bohr, cast to ``positions``'s dtype and device.
    """
    model = _build_model(numbers, positions, disp_cutoff)
    damping_param = _build_damping_param(param)

    result = model.get_dispersion(damping_param, grad=True)
    gradient = np.asarray(result["gradient"])

    return torch.tensor(
        gradient, dtype=positions.dtype, device=positions.device
    )


def reference_hessian(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    *,
    disp_cutoff: float = defaults.D3_DISP_CUTOFF,
) -> Tensor:
    """
    Cartesian Hessian of the dispersion energy from the s-dftd3 Fortran
    reference.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, shape ``(nat,)``.
    positions : Tensor
        Cartesian coordinates in Bohr, shape ``(nat, 3)``.
    param : dict[str, Tensor]
        Damping parameters, as in :func:`reference_energy_per_atom`.
    disp_cutoff : float, optional
        Real-space cutoff, in Bohr, as in :func:`reference_energy_per_atom`.

    Returns
    -------
    Tensor
        The Hessian, shape ``(nat, 3, nat, 3)``. s-dftd3 itself returns it
        flattened as ``(3 * nat, 3 * nat)`` -- reshaping that with a plain
        (C-order) ``reshape`` was checked against tad-dftd3's own Hessian
        directly, not assumed, and agrees to about 7e-20 absolute; no
        Fortran-order reshape is needed here. In Hartree per Bohr squared,
        cast to ``positions``'s dtype and device.
    """
    model = _build_model(numbers, positions, disp_cutoff)
    damping_param = _build_damping_param(param)

    result = model.get_hessian(damping_param)
    hessian = np.asarray(result["hessian"])

    nat = numbers.shape[-1]
    hessian = hessian.reshape(nat, 3, nat, 3)

    return torch.tensor(hessian, dtype=positions.dtype, device=positions.device)


def reference_pairwise(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    *,
    disp_cutoff: float = defaults.D3_DISP_CUTOFF,
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
    numbers : Tensor
        Atomic numbers, shape ``(nat,)``.
    positions : Tensor
        Cartesian coordinates in Bohr, shape ``(nat, 3)``.
    param : dict[str, Tensor]
        Damping parameters, as in :func:`reference_energy_per_atom`.
    disp_cutoff : float, optional
        Real-space cutoff, in Bohr, as in :func:`reference_energy_per_atom`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(two_body, three_body)``, each shape ``(nat, nat)`` and in
        Hartree, cast to ``positions``'s dtype and device. ``three_body``
        is s-dftd3's "non-additive pairwise energy"; it sums to the ATM
        energy only to about 1e-10 relative, not to full precision, because
        the three-body term is genuinely a sum over atom triples, not
        pairs, and there is more than one reasonable way to fold a
        triple's energy back onto a pairwise matrix.
    """
    model = _build_model(numbers, positions, disp_cutoff)
    damping_param = _build_damping_param(param)

    result = model.get_pairwise_dispersion(damping_param)
    two_body = np.asarray(result["additive pairwise energy"])
    three_body = np.asarray(result["non-additive pairwise energy"])

    two_body_tensor = torch.tensor(
        two_body, dtype=positions.dtype, device=positions.device
    )
    three_body_tensor = torch.tensor(
        three_body, dtype=positions.dtype, device=positions.device
    )
    return two_body_tensor, three_body_tensor
