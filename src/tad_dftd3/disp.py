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
Dispersion energy
=================

This module provides the dispersion energy evaluation for the pairwise interactions.

Example
-------
>>> import torch
>>> import tad_dftd3 as d3
>>> import tad_mctc as mctc
>>> numbers = torch.tensor([  # define fragments by setting atomic numbers to zero
...     [8, 1, 1, 8, 1, 6, 1, 1, 1],
...     [0, 0, 0, 8, 1, 6, 1, 1, 1],
...     [8, 1, 1, 0, 0, 0, 0, 0, 0],
... ])
>>> positions = torch.tensor([  # define coordinates once
...     [-4.224363834, +0.270465696, +0.527578960],
...     [-5.011768887, +1.780116228, +1.143194385],
...     [-2.468758653, +0.479766200, +0.982905589],
...     [+1.146167671, +0.452771215, +1.257722311],
...     [+1.841554378, -0.628298322, +2.538065200],
...     [+2.024899840, -0.438480095, -1.127412563],
...     [+1.210773578, +0.791908575, -2.550591723],
...     [+4.077073644, -0.342495506, -1.267841745],
...     [+1.404422261, -2.365753991, -1.503620411],
... ], dtype=torch.double).repeat(numbers.shape[0], 1, 1)
>>> ref = d3.reference.Reference(dtype=torch.double)
>>> param = dict( # r²SCAN-D3(BJ)
...     a1=torch.tensor(0.49484001, dtype=torch.double),
...     s8=torch.tensor(0.78981345, dtype=torch.double),
...     a2=torch.tensor(5.73083694, dtype=torch.double),
... )
>>> structure = mctc.Structure(numbers=numbers, positions=positions)
>>> cn_model = d3.ncoord.cn_d3.replace(cutoff=d3.defaults.D3_CN_CUTOFF)
>>> cn = cn_model(structure)
>>> weights = d3.model.weight_references(numbers, cn, ref)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> energy = d3.disp.dispersion(numbers, positions, param, c6)
>>> print(f"{torch.sum(energy[0] - energy[1] - energy[2]):.7f}")  # Hartree
-0.0003964
"""

from __future__ import annotations

from typing import Any

import torch
from tad_mctc import Structure, storch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.batch import real_pairs
from tad_mctc.data import pse
from tad_mctc.typing import (
    DD,
    CountingFunction,
    DampingFunction,
    TableFunction,
    Tensor,
)

from . import defaults, model, ncoord
from .cutoff import Cutoff
from .damping import dispersion_atm, rational_damping
from .data.table import element_table, reject_renamed_tables
from .model.weights import WeightingFunction
from .reference import Reference, _default_reference

__all__ = ["dftd3", "dispersion", "dispersion2", "dispersion3"]


def _resolve_cutoff(cutoff: Cutoff | None) -> Cutoff:
    """Default the caller's cutoffs; reject a single value."""
    if cutoff is None:
        return Cutoff()

    # Up to 0.6.0 one cutoff was shared by the two- and three-body term.
    # They now differ, as in s-dftd3, so a single value is ambiguous and
    # must not be reinterpreted silently.
    if not isinstance(cutoff, Cutoff):
        raise TypeError(
            "The 'cutoff' argument must be a 'tad_dftd3.cutoff.Cutoff' "
            f"instance, not '{type(cutoff).__name__}'. The parts of the D3 "
            "model use different real-space cutoffs, so a single value is "
            "ambiguous. Use e.g. 'Cutoff(disp2=60.0, disp3=40.0)'."
        )

    return cutoff


def _check_inputs(numbers: Tensor, positions: Tensor) -> None:
    """Reject inconsistent shapes and elements without D3 parameters."""
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    if not is_functorch_tensor(numbers):
        if torch.max(numbers) >= defaults.MAX_ELEMENT:
            raise ValueError(
                f"No D3 parameters available for Z > {defaults.MAX_ELEMENT-1} "
                f"({pse.Z2S[defaults.MAX_ELEMENT]})."
            )


@reject_renamed_tables
def dftd3(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor | float],
    *,
    ref: Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = model.gaussian_weight,
    damping_function: DampingFunction = rational_damping,
) -> Tensor:
    """
    Evaluate DFT-D3 dispersion energy for a batch of geometries.

    The element parameters `rcov_table`, `rvdw_table` and `r4r2_table` are
    tables indexed by atomic number (entry 0 is the dummy), not per-atom
    values, and must have the shape of their default table. Gradients with
    respect to them come out per element, summed over all atoms and systems.
    Up to 0.7.0 they were called `rcov`, `rvdw` and `r4r2` and took per-atom
    values; the old names are rejected with a :class:`TypeError`.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers of the atoms in the system, of shape ``(nat,)`` for a
        single system or ``(nbatch, nat)`` for a batch (padded with zeros,
        e.g. by :func:`tad_mctc.batch.pack`).
    positions : torch.Tensor
        Cartesian coordinates of the atoms in the system, of shape
        ``(nat, 3)`` or ``(nbatch, nat, 3)``, matching `numbers`.
    param : dict[str, Tensor | float]
        DFT-D3 damping parameters. The three-body term is skipped if `s9` is
        missing or zero; see :func:`dispersion` for when that is decided.
    ref : reference.Reference, optional
        Reference C6 coefficients.
    rcov_table : Tensor | TableFunction, optional
        Covalent radii per element, of shape ``(119,)``. Defaults to
        :func:`tad_mctc.data.radii.COV_D3`. Passed to the coordination
        number model, :data:`tad_mctc.ncoord.cn_d3`.
    rvdw_table : Tensor | TableFunction, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
        Defaults to :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
        Defaults to :func:`tad_dftd3.data.R4R2`.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model. Defaults to
        :class:`tad_dftd3.cutoff.Cutoff`.
    damping_function : Callable, optional
        Damping function evaluate distance dependent contributions.
    weighting_function : Callable, optional
        Function to calculate weight of individual reference systems.
    counting_function : Callable, optional
        Calculates counting value in range 0 to 1 for each atom pair.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D3 dispersion energy for each geometry, of the
        shape of `numbers`.

    Raises
    ------
    ValueError
        If the shapes of `numbers` and `positions` are not consistent, an
        element without D3 parameters is present, or `rcov_table`,
        `rvdw_table` or `r4r2_table` is not shaped like a table.
    TypeError
        If one of the names of 0.7.0, `rcov`, `rvdw` or `r4r2`, is passed.
    """
    _check_inputs(numbers, positions)

    cutoff = _resolve_cutoff(cutoff)
    if ref is None:
        # `dftd3` only reads the reference, so it need not be copied.
        ref = _default_reference(positions)

    cn_model = ncoord.cn_d3.replace(
        count=counting_function,
        cutoff=cutoff.cn,
        rcov=element_table(rcov_table, "rcov_table", positions),
    )
    cn = cn_model(Structure(numbers=numbers, positions=positions))
    weights = model.weight_references(numbers, cn, ref, weighting_function)
    c6 = model.atomic_c6(numbers, weights, ref)

    return dispersion(
        numbers,
        positions,
        param,
        c6,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        damping_function=damping_function,
        cutoff=cutoff,
    )


@reject_renamed_tables
def dispersion(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: Cutoff | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    As in :func:`dftd3`, the element parameters `rvdw_table` and
    `r4r2_table` are tables indexed by atomic number, not per-atom values.

    The three-body term is skipped if `param` has no ``"s9"`` or it is zero
    (see :func:`_has_three_body`). To skip it under ``torch.compile``, give
    `s9` as a Python number (``0.0``) or leave it out.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor | float]
        DFT-D3 damping parameters. `s9` may be a Python number.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw_table : Tensor | TableFunction, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
        Defaults to :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
        Defaults to :func:`tad_dftd3.data.R4R2`.
    damping_function : Callable
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model. Defaults to
        :class:`tad_dftd3.cutoff.Cutoff`.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D3 dispersion energy for each geometry.

    Raises
    ------
    ValueError
        If the shapes of `numbers` and `positions` are not consistent, an
        element without D3 parameters is present, or `rvdw_table` or
        `r4r2_table` does not have the shape of its default table.
    TypeError
        If one of the names of 0.7.0, `rvdw` or `r4r2`, is passed.
    """
    cutoff = _resolve_cutoff(cutoff)

    _check_inputs(numbers, positions)

    # Resolved once here and passed on as tensors. Also rejects a wrong
    # `rvdw_table` if the three-body term, its only user, is not evaluated.
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)

    # two-body dispersion
    energy = dispersion2(
        numbers,
        positions,
        param,
        c6,
        r4r2_table=r4r2,
        damping_function=damping_function,
        cutoff=cutoff.disp2,
        **kwargs,
    )

    # three-body dispersion
    #
    # Not added in place: under `vmap` over `s9`, only the three-body term is
    # batched, and the two-body energy cannot take its batch dimension.
    if "s9" in param and _has_three_body(param["s9"]):
        e3 = dispersion3(
            numbers,
            positions,
            param,
            c6,
            rvdw_table=rvdw,
            cutoff=cutoff.disp3,
        )
        energy = energy + e3

    return energy


def _has_three_body(s9: Tensor | float | int) -> bool:
    """
    Whether the three-body term must be evaluated for the scaling `s9`.

    A Python number is a constant, also to ``torch.compile``, and the term
    is skipped for zero. A tensor is skipped only for ``s9 == 0`` in plain
    eager mode. A tensor that is traced (``torch.compile``, which cannot
    branch on it), batched (``vmap``) or differentiated (autograd or
    ``torch.func``) needs the term even at zero: the energy is linear in
    `s9`, so its derivative with respect to `s9` is the three-body energy
    itself.
    """
    if not isinstance(s9, Tensor):
        return s9 != 0.0

    # `is_functorch_tensor` is also `True` while `torch.compile` traces, but
    # does not cover plain autograd, hence `requires_grad`.
    return is_functorch_tensor(s9) or s9.requires_grad or bool(s9 != 0.0)


@reject_renamed_tables
def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    r4r2_table: Tensor | TableFunction | None = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: float = defaults.D3_DISP2_CUTOFF,
    **kwargs: Any,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor | float]
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``, or
        ``None`` for :func:`tad_dftd3.data.R4R2`.
    damping_function : Callable, optional
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    cutoff : float, optional
        Real-space cutoff of the pairs, in Bohr. Defaults to
        :data:`tad_dftd3.defaults.D3_DISP2_CUTOFF`.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    r4r2 = element_table(r4r2_table, "r4r2_table", positions)
    r4r2_atom = r4r2[numbers]

    # Padding atoms look up the dummy entry 0 of the table, `r4r2 == 0`, and
    # the gradient of `sqrt(qq)` in the damping function would then be
    # infinite. Masking only the output does not help (0 * inf = NaN), so the
    # input is replaced too, as is done for the distances above.
    r4r2_atom = torch.where(numbers != 0, r4r2_atom, torch.ones_like(r4r2_atom))
    qq = 3 * r4r2_atom.unsqueeze(-1) * r4r2_atom.unsqueeze(-2)
    c8 = c6 * qq

    keep = mask * (distances <= cutoff)
    zero = torch.tensor(0.0, **dd)
    t6 = torch.where(
        keep, damping_function(6, distances, qq, param, **kwargs), zero
    )
    t8 = torch.where(
        keep, damping_function(8, distances, qq, param, **kwargs), zero
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))
    return s6 * e6 + s8 * e8


@reject_renamed_tables
def dispersion3(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    rvdw_table: Tensor | TableFunction | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    rs9: Tensor | float | None = None,
) -> Tensor:
    """
    Three-body dispersion term. Currently this is only a wrapper for the
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor | float]
        Dictionary of dispersion parameters. Default values are used for
        missing keys.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``, or
        ``None`` for :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    cutoff : float, optional
        Real-space cutoff, in Bohr. Defaults to
        :data:`tad_dftd3.defaults.D3_DISP3_CUTOFF`.
    rs9 : Tensor | float, optional
        Scaling for van-der-Waals radii in damping function. Defaults to `4.0/3.0`.

    Returns
    -------
    Tensor
        Atom-resolved three-body dispersion energy.
    """
    return dispersion_atm(
        numbers,
        positions,
        c6,
        rvdw_table=rvdw_table,
        cutoff=cutoff,
        s9=param.get("s9"),
        rs9=rs9,
        alp=param.get("alp"),
    )
