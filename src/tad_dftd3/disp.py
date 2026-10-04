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
>>> energy = d3.disp.dispersion(structure, param, c6)
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
from tad_mctc.neighbor.images import (
    PeriodicShifts,
    build_periodic_shifts,
    wrap_to_central_cell,
)
from tad_mctc.typing import (
    DD,
    CountingFunction,
    DampingFunction,
    TableFunction,
    Tensor,
)

from . import defaults, model, ncoord
from ._checks import takes_structure
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


def _check_inputs(structure: Structure, shifts: PeriodicShifts | None) -> None:
    """
    Reject elements without D3 parameters, and an inconsistent cell.

    `Structure` itself checks the shapes of its fields when it is built, but
    not that each system of a batch has its own lattice and periodic mask:
    ``vmap`` over a batched `Structure` splits every field along its first
    dimension, so a single ``(3, 3)`` lattice (or ``(3,)`` mask) shared by
    the batch would arrive as one of its rows (or entries).
    """
    numbers = structure.numbers
    if not is_functorch_tensor(numbers):
        if torch.max(numbers) >= defaults.MAX_ELEMENT:
            raise ValueError(
                f"No D3 parameters available for Z > {defaults.MAX_ELEMENT-1} "
                f"({pse.Z2S[defaults.MAX_ELEMENT]})."
            )

    lattice = structure.lattice
    if lattice is None:
        if shifts is not None:
            raise ValueError(
                "'shifts' are periodic image shifts, but 'structure' has no "
                "'lattice'."
            )
        return

    # `Structure` fills in a mask whenever it has a lattice.
    periodic = structure.periodic
    assert periodic is not None
    if lattice.shape[-2:] != (3, 3) or periodic.shape[-1:] != (3,):
        raise ValueError(
            "The lattice and the periodic mask must have shapes "
            f"'(..., 3, 3)' and '(..., 3)', not '{tuple(lattice.shape)}' "
            f"and '{tuple(periodic.shape)}'. Under `vmap` over a batched "
            "'Structure', give each system its own lattice and mask, e.g. "
            "'lattice.expand(nbatch, 3, 3)' and "
            "'periodic.expand(nbatch, 3)'."
        )


def _cell_shifts(
    cell: Structure, shifts: PeriodicShifts | None, cutoff: float
) -> PeriodicShifts:
    """
    The periodic image shifts to sum over at `cutoff`: those given, after
    checking that they cover `cell` at `cutoff`, or else built here.

    Building the table reads the values of the lattice (how many images the
    cutoff reaches), so it is done eagerly, outside the graph. Under
    ``torch.compile``, ``vmap`` or ``jacrev`` with respect to the lattice, a
    table must be given instead (see :func:`dftd3`).
    """
    if shifts is not None:
        shifts.check_compatible(cell, cutoff)
        return shifts

    # `Structure` fills in a mask whenever it has a lattice.
    assert cell.lattice is not None and cell.periodic is not None
    return build_periodic_shifts(cell.lattice, cell.periodic, cutoff)


def _check_three_body(
    param: dict[str, Tensor | float], structure: Structure
) -> None:
    """
    Reject a three-body term for a cell, which has no periodic evaluation.

    A dense periodic ATM term would sum over pairs of images per triple,
    ``n_shift**2`` times the memory of the molecular one.
    """
    if structure.lattice is None:
        return

    if "s9" in param and _has_three_body(param["s9"]):
        raise ValueError(
            "The three-body (ATM) term has no periodic evaluation; leave "
            "out 's9' or set it to the Python number 0.0 for a cell. A "
            "tensor 's9' always counts as nonzero under `torch.compile`, "
            "`vmap` and autograd, even if it is zero."
        )


@reject_renamed_tables
@takes_structure
def dftd3(
    structure: Structure,
    param: dict[str, Tensor | float],
    *,
    shifts: PeriodicShifts | None = None,
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

    If `structure` has a ``lattice``, it is a periodic cell, along the axes
    of its ``periodic`` mask: the coordination number and the two-body
    energy sum over every periodic image within their cutoff, as in
    s-dftd3. Atoms need not lie inside the cell. The three-body term has no
    periodic evaluation, so for a cell `s9` must be missing or the Python
    number ``0.0``.

    Which periodic images lie within a cutoff depends on the values of the
    lattice, so by default the image shifts are built eagerly on each call.
    Under ``torch.compile(fullgraph=True)``, ``vmap`` over cells, or
    ``jacrev``/``jacfwd`` with respect to the lattice, build them once
    beforehand and pass them as `shifts`, at the larger of the
    coordination-number and two-body cutoffs::

        from tad_mctc.neighbor.images import build_periodic_shifts

        cutoff = Cutoff()
        shifts = build_periodic_shifts(
            structure.lattice, structure.periodic, max(cutoff.cn, cutoff.disp2)
        )

    To differentiate with respect to the positions or the lattice with
    ``torch.func``, replace them in the structure inside the function, e.g.
    ``jacrev(lambda pos: dftd3(structure.replace(positions=pos), param))``.

    Parameters
    ----------
    structure : Structure
        The system: atomic numbers, of shape ``(nat,)``, and Cartesian
        coordinates in Bohr, of shape ``(nat, 3)``; for a batch with a
        leading ``nbatch`` dimension, padded with zeros (e.g. by
        :func:`tad_mctc.io.structure.pack_structures`). A periodic cell
        also has a ``lattice`` (vectors as rows, in Bohr, ``(3, 3)`` or
        ``(nbatch, 3, 3)``) and its ``periodic`` axes.
    param : dict[str, Tensor | float]
        DFT-D3 damping parameters. The three-body term is skipped if `s9` is
        missing or zero; see :func:`dispersion` for when that is decided.
    shifts : PeriodicShifts | None, optional
        Periodic image shifts from
        :func:`tad_mctc.neighbor.images.build_periodic_shifts`, at least at
        the coordination-number and the two-body cutoff. Defaults to
        building them on each call. A table covering more images than
        needed gives the same energy.
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
        shape of ``structure.numbers``.

    Raises
    ------
    ValueError
        If an element without D3 parameters is present, `rcov_table`,
        `rvdw_table` or `r4r2_table` is not shaped like a table, `shifts`
        is given for a molecule or does not cover the cell at the cutoffs,
        or a cell has a three-body term.
    TypeError
        If `structure` is not a ``Structure``, or one of the names of
        0.7.0, `rcov`, `rvdw` or `r4r2`, is passed.
    """
    _check_inputs(structure, shifts)
    _check_three_body(param, structure)

    numbers, positions = structure.numbers, structure.positions

    cutoff = _resolve_cutoff(cutoff)
    if ref is None:
        # `dftd3` only reads the reference, so it need not be copied.
        ref = _default_reference(positions)

    cn_model = ncoord.cn_d3.replace(
        count=counting_function,
        cutoff=cutoff.cn,
        rcov=element_table(rcov_table, "rcov_table", positions),
    )
    if structure.lattice is None:
        cn = cn_model(structure)
    else:
        cn = cn_model(structure, _cell_shifts(structure, shifts, cutoff.cn))
    weights = model.weight_references(numbers, cn, ref, weighting_function)
    c6 = model.atomic_c6(numbers, weights, ref)

    return dispersion(
        structure,
        param,
        c6,
        shifts=shifts,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        damping_function=damping_function,
        cutoff=cutoff,
    )


@reject_renamed_tables
@takes_structure
def dispersion(
    structure: Structure,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    shifts: PeriodicShifts | None = None,
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
    structure : Structure
        The system, a molecule or a periodic cell, see :func:`dftd3`.
    param : dict[str, Tensor | float]
        DFT-D3 damping parameters. `s9` may be a Python number.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    shifts : PeriodicShifts | None, optional
        Periodic image shifts, at least at the two-body cutoff, see
        :func:`dftd3`.
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
        If an element without D3 parameters is present, `rvdw_table` or
        `r4r2_table` does not have the shape of its default table, or the
        cell is not valid (see :func:`dftd3`).
    TypeError
        If `structure` is not a ``Structure``, or one of the names of
        0.7.0, `rvdw` or `r4r2`, is passed.
    """
    cutoff = _resolve_cutoff(cutoff)

    _check_inputs(structure, shifts)
    _check_three_body(param, structure)

    positions = structure.positions

    # Resolved once here and passed on as tensors. Also rejects a wrong
    # `rvdw_table` if the three-body term, its only user, is not evaluated.
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)

    # two-body dispersion
    energy = dispersion2(
        structure,
        param,
        c6,
        shifts=shifts,
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
            structure,
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
@takes_structure
def dispersion2(
    structure: Structure,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    shifts: PeriodicShifts | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    damping_function: DampingFunction = rational_damping,
    cutoff: float = defaults.D3_DISP2_CUTOFF,
    **kwargs: Any,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    For a periodic cell (`structure` has a ``lattice``), every atom is
    paired with every periodic image of every atom within the `cutoff`,
    including the images of the atom itself. This takes
    ``O(nat**2 * n_shift)`` memory.

    Parameters
    ----------
    structure : Structure
        The system, a molecule or a periodic cell, see :func:`dftd3`.
    param : dict[str, Tensor | float]
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    shifts : PeriodicShifts | None, optional
        Periodic image shifts, at least at `cutoff`, see :func:`dftd3`.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``, or
        ``None`` for :func:`tad_dftd3.data.R4R2`.
    damping_function : Callable, optional
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
    cutoff : float, optional
        Real-space cutoff of the pairs, in Bohr. Defaults to
        :data:`tad_dftd3.defaults.D3_DISP2_CUTOFF`.

    Raises
    ------
    ValueError
        If the cell is not valid (see :func:`dftd3`).
    TypeError
        If `structure` is not a ``Structure``.
    """
    _check_inputs(structure, shifts)

    numbers, positions = structure.numbers, structure.positions
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    r4r2 = element_table(r4r2_table, "r4r2_table", positions)
    r4r2_atom = r4r2[numbers]

    # Padding atoms look up the dummy entry 0 of the table, `r4r2 == 0`, and
    # the gradient of `sqrt(qq)` in the damping function would then be
    # infinite. Masking only the output does not help (0 * inf = NaN), so the
    # input is replaced too, as is done for the distances.
    r4r2_atom = torch.where(numbers != 0, r4r2_atom, torch.ones_like(r4r2_atom))
    qq = 3 * r4r2_atom.unsqueeze(-1) * r4r2_atom.unsqueeze(-2)

    if structure.lattice is None:
        distances, keep = _molecular_distances(numbers, positions, cutoff)
        atom_pair_dims: tuple[int, ...] = (-1,)
    else:
        image_shifts = _cell_shifts(structure, shifts, cutoff)
        distances, keep = _periodic_distances(structure, image_shifts, cutoff)

        # One entry per pair and image: the pair's C6 and `qq` apply to
        # every image alike.
        c6 = c6.unsqueeze(-1)
        qq = qq.unsqueeze(-1)
        atom_pair_dims = (-2, -1)

    c8 = c6 * qq

    zero = torch.tensor(0.0, **dd)
    t6 = torch.where(
        keep, damping_function(6, distances, qq, param, **kwargs), zero
    )
    t8 = torch.where(
        keep, damping_function(8, distances, qq, param, **kwargs), zero
    )

    e6 = -0.5 * torch.sum(c6 * t6, dim=atom_pair_dims)
    e8 = -0.5 * torch.sum(c8 * t8, dim=atom_pair_dims)

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))
    return s6 * e6 + s8 * e8


def _molecular_distances(
    numbers: Tensor, positions: Tensor, cutoff: float
) -> tuple[Tensor, Tensor]:
    """
    Distances between all atoms of a molecule, ``(..., nat, nat)``, and
    which of them are pairs within `cutoff`.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    keep = mask * (distances <= cutoff)
    return distances, keep


def _periodic_distances(
    cell: Structure, shifts: PeriodicShifts, cutoff: float
) -> tuple[Tensor, Tensor]:
    """
    Distances from every atom of a cell to every periodic image of every
    atom, ``(..., nat, nat, n_shift)``, and which of them are pairs within
    `cutoff`. Entry ``(i, j, image)`` is the distance from atom ``i`` to
    that image of atom ``j``.

    Differentiable in the positions and the lattice to any order: the shift
    table is integer data, and folding into the central cell only adds
    whole lattice vectors.
    """
    assert cell.lattice is not None and cell.periodic is not None
    lattice = cell.lattice

    # The shifts reach every image within the cutoff only from atoms inside
    # the central cell (see `build_periodic_shifts`), so the atoms are
    # folded into it first, as s-dftd3 does. The mask gets an explicit atom
    # axis, so that a batched `(nbatch, 3)` mask cannot align with the atom
    # axis when the batch is as large as the number of atoms.
    positions, _ = wrap_to_central_cell(
        cell.positions, lattice, cell.periodic.unsqueeze(-2)
    )

    # (..., n_shift, 3), one table for every system of a batch
    translations = shifts.shifts.to(positions.dtype) @ lattice

    # (..., nat, nat, 3): entry `(i, j)` points from atom `i` to atom `j`
    pair_vectors = positions.unsqueeze(-3) - positions.unsqueeze(-2)

    # (..., nat, nat, n_shift, 3)
    image_translations = translations[..., None, None, :, :]
    image_vectors = pair_vectors.unsqueeze(-2) + image_translations
    distance_squared = torch.sum(image_vectors * image_vectors, dim=-1)

    is_pair = _periodic_pairs(cell.numbers, shifts.shifts, cell.periodic)
    keep = is_pair & (distance_squared <= cutoff * cutoff)

    # Replaced before the square root, not after: an atom with itself at
    # the zero shift has a distance of zero, where the derivative of the
    # square root is infinite and would turn the masked gradient into NaN.
    one = torch.ones_like(distance_squared)
    distances = torch.sqrt(torch.where(keep, distance_squared, one))

    return distances, keep


def _periodic_pairs(
    numbers: Tensor, shifts: Tensor, periodic: Tensor
) -> Tensor:
    """
    Which entries ``(i, j, image)`` of the periodic pair grid are pairs,
    ``(..., nat, nat, n_shift)``: both atoms are real (not padding), the
    entry is not an atom with itself at the zero shift, and the image does
    not lie along an axis that is not periodic for this system.

    An atom with one of its own images at a non-zero shift is a pair.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers, of shape ``(..., nat)``, zero for padding.
    shifts : Tensor
        Integer lattice translations, of shape ``(n_shift, 3)``.
    periodic : Tensor
        Periodic axes of each system, of shape ``(3,)`` or ``(..., 3)``.
    """
    both_real = real_pairs(numbers, mask_diagonal=False).unsqueeze(-1)

    nat = numbers.shape[-1]
    is_same_atom = torch.eye(nat, dtype=torch.bool, device=numbers.device)
    is_zero_shift = (shifts == 0).all(dim=-1)
    is_self_pair = is_same_atom.unsqueeze(-1) & is_zero_shift

    # One table serves a whole batch, so it may translate along an axis that
    # is periodic for another system but not for this one. The cutoff does
    # not remove such an image: a non-periodic axis may have a short
    # placeholder lattice vector.
    along_open_axis = (shifts != 0) & ~periodic.unsqueeze(-2)
    is_image = ~along_open_axis.any(dim=-1)  # (..., n_shift)
    is_image = is_image.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, n_shift)

    return both_real & ~is_self_pair & is_image


@reject_renamed_tables
@takes_structure
def dispersion3(
    structure: Structure,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    rvdw_table: Tensor | TableFunction | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    rs9: Tensor | float | None = None,
) -> Tensor:
    """
    Three-body dispersion term. Currently this is only a wrapper for the
    Axilrod-Teller-Muto dispersion term, which has no periodic evaluation.

    Parameters
    ----------
    structure : Structure
        The system, a molecule (see :func:`dftd3`).
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

    Raises
    ------
    ValueError
        If `structure` is a periodic cell.
    TypeError
        If `structure` is not a ``Structure``.
    """
    return dispersion_atm(
        structure,
        c6,
        rvdw_table=rvdw_table,
        cutoff=cutoff,
        s9=param.get("s9"),
        rs9=rs9,
        alp=param.get("alp"),
    )
