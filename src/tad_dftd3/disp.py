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
from tad_mctc.ncoord.common import _periodic_images, sum_over_neighborlist
from tad_mctc.neighbor import pair_distance_squared, split_lattice
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.neighbor.list import NeighborList, build_neighborlists
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


def _check_pairs(
    structure: Structure,
    shifts: PeriodicShifts | None,
    nbl: NeighborList | None,
    cutoff: float,
) -> None:
    """
    Reject a neighbour list that is given together with `shifts`, or that
    does not cover `structure` at `cutoff`.

    The check reads the values of the periodic mask, so it is eager: with
    ``torch.compile``, ``vmap`` or ``jacrev``, call it once beforehand (any
    evaluation does) and pass the list in.
    """
    if nbl is None:
        return

    if shifts is not None:
        raise ValueError(
            "Give either periodic image 'shifts' or a neighbour list, not "
            "both: they are two ways of enumerating the same pairs."
        )

    nbl.check_compatible(structure, cutoff)


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
    nbl_cn: NeighborList | None = None,
    nbl_disp2: NeighborList | None = None,
    nbl_disp3: NeighborList | None = None,
    sparse: bool = False,
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

    The coordination number also runs over every shift of this table, so with
    the default cutoffs (40 and 60 Bohr) it does about three times the work
    it would with a table built at its own cutoff. Only the energy is
    unaffected.

    Instead of dense, all-pairs sums, the coordination number and the two-body
    energy can run over pre-built, padded neighbour lists
    (:class:`tad_mctc.neighbor.list.NeighborList`), which scale linearly with
    the number of atoms, for molecules, batches and cells alike. With
    ``sparse=True`` both lists are built eagerly on each call, sharing one
    search. To reuse them, e.g. over the steps of a molecular dynamics, build
    them once with :func:`tad_mctc.neighbor.list.build_neighborlists`, each at
    (at least) its own cutoff, and pass them as `nbl_cn` and `nbl_disp2`::

        from tad_mctc.neighbor.list import build_neighborlists

        cutoff = Cutoff()
        nbl_cn, nbl_disp2 = build_neighborlists(
            structure, (cutoff.cn, cutoff.disp2), skin=1.0
        )

    A list built at a larger cutoff, or with a skin, gives the same energy.
    The three-body term is evaluated from a list only if one is given as
    `nbl_disp3`, see :func:`dispersion3`.

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
        needed gives the same energy. Not together with a neighbour list.
    nbl_cn : NeighborList | None, optional
        Neighbour list for the coordination number, built at least at its
        cutoff. Defaults to the dense (or `shifts`) evaluation, or to a list
        built here if `sparse`.
    nbl_disp2 : NeighborList | None, optional
        Neighbour list for the two-body energy, built at least at its
        cutoff. Defaults like `nbl_cn`.
    nbl_disp3 : NeighborList | None, optional
        Neighbour list for the three-body term of a molecule, built at least
        at its cutoff. Never built by `sparse`: the memory of the sparse
        three-body term grows steeply with its cutoff (see
        :func:`dispersion3`), so choosing it, and a `cutoff.disp3` to match,
        is left to the caller.
    sparse : bool, optional
        Build the neighbour lists that are not given, instead of evaluating
        that term densely. Defaults to ``False``.
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
        a neighbour list does not match the structure or its cutoff, or is
        given together with `shifts`, or a cell has a three-body term.
    TypeError
        If `structure` is not a ``Structure``, or one of the names of
        0.7.0, `rcov`, `rvdw` or `r4r2`, is passed.
    """
    _check_inputs(structure, shifts)
    _check_three_body(param, structure)

    numbers, positions = structure.numbers, structure.positions

    cutoff = _resolve_cutoff(cutoff)

    if sparse:
        if shifts is not None:
            raise ValueError(
                "'sparse' builds neighbour lists, which replace the periodic "
                "image 'shifts'; give only one of them."
            )

        # One shared search for the lists that are missing, eagerly: it is
        # data dependent. Both are built if only one is missing, which costs
        # no more than the larger one alone.
        if nbl_cn is None or nbl_disp2 is None:
            built_cn, built_disp2 = build_neighborlists(
                structure, (cutoff.cn, cutoff.disp2)
            )
            nbl_cn = built_cn if nbl_cn is None else nbl_cn
            nbl_disp2 = built_disp2 if nbl_disp2 is None else nbl_disp2

    _check_pairs(structure, shifts, nbl_cn, cutoff.cn)
    _check_pairs(structure, shifts, nbl_disp2, cutoff.disp2)
    if ref is None:
        # `dftd3` only reads the reference, so it need not be copied.
        ref = _default_reference(positions)

    cn_model = ncoord.cn_d3.replace(
        count=counting_function,
        cutoff=cutoff.cn,
        rcov=element_table(rcov_table, "rcov_table", positions),
    )
    # Builds the image shifts of a cell at its own cutoff, or checks the
    # given ones against it.
    cn = cn_model(structure, nbl_cn if nbl_cn is not None else shifts)
    weights = model.weight_references(numbers, cn, ref, weighting_function)
    c6 = model.atomic_c6(numbers, weights, ref)

    # The inputs are checked above, so the unchecked kernel is called.
    return _dispersion(
        structure,
        param,
        c6,
        shifts=shifts,
        nbl=nbl_disp2,
        nbl_disp3=nbl_disp3,
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
    nbl: NeighborList | None = None,
    nbl_disp3: NeighborList | None = None,
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
    nbl : NeighborList | None, optional
        Neighbour list for the two-body energy, built at least at its
        cutoff, instead of the dense evaluation, see :func:`dftd3`. Not
        together with `shifts`.
    nbl_disp3 : NeighborList | None, optional
        Neighbour list for the three-body term, built at least at its
        cutoff, see :func:`dispersion3`. A molecule only.
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
    _check_inputs(structure, shifts)
    _check_three_body(param, structure)

    cutoff = _resolve_cutoff(cutoff)
    _check_pairs(structure, shifts, nbl, cutoff.disp2)

    return _dispersion(
        structure,
        param,
        c6,
        shifts=shifts,
        nbl=nbl,
        nbl_disp3=nbl_disp3,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        damping_function=damping_function,
        cutoff=cutoff,
        **kwargs,
    )


def _dispersion(
    structure: Structure,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    shifts: PeriodicShifts | None,
    nbl: NeighborList | None,
    nbl_disp3: NeighborList | None,
    rvdw_table: Tensor | TableFunction | None,
    r4r2_table: Tensor | TableFunction | None,
    damping_function: DampingFunction,
    cutoff: Cutoff,
    **kwargs: Any,
) -> Tensor:
    """
    :func:`dispersion` without checking its inputs, for callers that have
    already checked them.
    """
    positions = structure.positions

    # Resolved once here and passed on as tensors. Also rejects a wrong
    # `rvdw_table` if the three-body term, its only user, is not evaluated.
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)

    # two-body dispersion
    energy = _dispersion2(
        structure,
        param,
        c6,
        shifts=shifts,
        nbl=nbl,
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
            nbl=nbl_disp3,
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
    nbl: NeighborList | None = None,
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
    nbl : NeighborList | None, optional
        Neighbour list, built at least at `cutoff`, to sum over instead of
        the dense evaluation, see :func:`dftd3`. Not together with `shifts`.
        This takes ``O(n_pairs)`` memory instead of ``O(nat**2 * n_shift)``.
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
    _check_pairs(structure, shifts, nbl, cutoff)

    return _dispersion2(
        structure,
        param,
        c6,
        shifts=shifts,
        nbl=nbl,
        r4r2_table=r4r2_table,
        damping_function=damping_function,
        cutoff=cutoff,
        **kwargs,
    )


def _dispersion2(
    structure: Structure,
    param: dict[str, Tensor | float],
    c6: Tensor,
    *,
    shifts: PeriodicShifts | None,
    nbl: NeighborList | None,
    r4r2_table: Tensor | TableFunction | None,
    damping_function: DampingFunction,
    cutoff: float,
    **kwargs: Any,
) -> Tensor:
    """
    :func:`dispersion2` without checking its inputs, for callers that have
    already checked them.
    """
    numbers, positions = structure.numbers, structure.positions
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    r4r2 = element_table(r4r2_table, "r4r2_table", positions)
    r4r2_atom = r4r2[numbers]

    # Padding atoms look up the dummy entry 0 of the table, `r4r2 == 0`, and
    # the gradient of `sqrt(qq)` in the damping function would then be
    # infinite. Masking only the output does not help (0 * inf = NaN), so the
    # input is replaced too, as is done for the distances.
    r4r2_atom = torch.where(numbers != 0, r4r2_atom, torch.ones_like(r4r2_atom))

    if nbl is not None:
        return _sparse_dispersion2(
            structure,
            param,
            c6,
            r4r2_atom,
            nbl,
            damping_function=damping_function,
            cutoff=cutoff,
            **kwargs,
        )

    qq = 3 * r4r2_atom.unsqueeze(-1) * r4r2_atom.unsqueeze(-2)

    if structure.lattice is None:
        distances, keep = _molecular_distances(numbers, positions, cutoff)
        qq_pairs = qq
    else:
        image_shifts = _cell_shifts(structure, shifts, cutoff)
        distances, keep = _periodic_distances(structure, image_shifts, cutoff)

        # One entry per pair and image, all with the pair's `qq`.
        qq_pairs = qq.unsqueeze(-1)

    zero = torch.tensor(0.0, **dd)
    t6 = torch.where(
        keep, damping_function(6, distances, qq_pairs, param, **kwargs), zero
    )
    t8 = torch.where(
        keep, damping_function(8, distances, qq_pairs, param, **kwargs), zero
    )

    if structure.lattice is not None:
        # C6 and C8 are the same for every image of a pair, so the damped
        # terms are summed over the images first and multiplied per pair.
        t6 = t6.sum(dim=-1)
        t8 = t8.sum(dim=-1)

    c8 = c6 * qq
    e6 = -0.5 * torch.sum(c6 * t6, dim=-1)
    e8 = -0.5 * torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))
    return s6 * e6 + s8 * e8


def _sparse_dispersion2(
    structure: Structure,
    param: dict[str, Tensor | float],
    c6: Tensor,
    r4r2_atom: Tensor,
    nbl: NeighborList,
    *,
    damping_function: DampingFunction,
    cutoff: float,
    **kwargs: Any,
) -> Tensor:
    """
    Two-body energy summed over a :class:`~tad_mctc.neighbor.list.NeighborList`
    instead of all pairs and images, for a molecule, a batch or a cell.

    A batch is one flat system: atom ``i`` of system ``b`` is ``b * nat + i``
    in the list, and the C6 of a pair is ``c6.reshape(-1, nat)[a, b % nat]``.
    Each entry of the list stands for the pair in both directions and the
    walk adds its contribution to both atoms, also for an atom with its own
    image, so every entry carries the ``-0.5`` of the dense sum.

    The walk is :func:`tad_mctc.ncoord.common.sum_over_neighborlist`, the one
    behind the sparse coordination number, with its conventions: padded slots
    point at a phantom atom appended to the positions, and all of it is
    fixed-shape tensor algebra. The positions are the ones as given, not
    folded into the cell: the shifts of the list include the fold.
    """
    positions = structure.positions
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    nat = positions.shape[-2]
    flat_positions = positions.reshape(-1, 3)
    total_atoms = flat_positions.shape[0]

    padded_positions = torch.cat(
        [flat_positions, flat_positions.new_zeros(1, 3)]
    )
    shared_lattice, system_lattices = split_lattice(structure.lattice)
    if system_lattices is not None:
        system_lattices = torch.cat(
            [system_lattices, system_lattices.new_zeros(1, 3, 3)]
        )

    c6_rows = c6.reshape(-1, nat)
    r4r2_flat = r4r2_atom.reshape(-1)

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))

    def pair_contributions(
        idx_i: Tensor, idx_j: Tensor, mask: Tensor, shift: Tensor
    ) -> tuple[Tensor, Tensor]:
        distance_squared = pair_distance_squared(
            idx_i,
            idx_j,
            shift,
            padded_positions,
            shared_lattice=shared_lattice,
            system_lattices=system_lattices,
            atoms_per_system=nat,
        )

        # The list may be wider than `cutoff` (skin, or built at a larger
        # cutoff for reuse), so its own mask is not enough. Replaced before
        # the square root, like the dense path: a padded slot is `sqrt(0)`
        # otherwise, and its derivative is infinite.
        mask = mask & (distance_squared <= cutoff * cutoff)
        distances = torch.sqrt(torch.where(mask, distance_squared, 1.0))

        # The tables know nothing about the phantom atom, so padded slots
        # are clamped to a real index, and masked below.
        real_i = idx_i.clamp(max=total_atoms - 1)
        real_j = idx_j.clamp(max=total_atoms - 1)
        c6_pair = c6_rows[real_i, real_j % nat]
        qq = 3 * r4r2_flat[real_i] * r4r2_flat[real_j]

        t6 = damping_function(6, distances, qq, param, **kwargs)
        t8 = damping_function(8, distances, qq, param, **kwargs)

        contribution = -0.5 * c6_pair * (s6 * t6 + s8 * qq * t8)
        contribution = torch.where(
            mask, contribution, torch.zeros_like(contribution)
        )
        return contribution, contribution

    energy = sum_over_neighborlist(nbl, pair_contributions, flat_positions)
    return energy.reshape(structure.numbers.shape)


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

    # The same images as the coordination number of tad-mctc: the atoms
    # folded into the central cell (which the shifts are built for, as in
    # s-dftd3), the Cartesian translation of each image, and which
    # `(i, j, image)` entries are pairs.
    images = _periodic_images(
        cell.numbers, cell.positions, cell.lattice, shifts.shifts, cell.periodic
    )
    assert images.translations is not None
    positions = images.positions

    # (..., nat, nat, 3): entry `(i, j)` points from atom `i` to atom `j`
    pair_vectors = positions.unsqueeze(-3) - positions.unsqueeze(-2)

    # (..., nat, nat, n_shift, 3)
    image_translations = images.translations[..., None, None, :, :]
    image_vectors = pair_vectors.unsqueeze(-2) + image_translations
    distance_squared = torch.sum(image_vectors * image_vectors, dim=-1)

    keep = images.valid & (distance_squared <= cutoff * cutoff)

    # Replaced before the square root, not after: an atom with itself at
    # the zero shift has a distance of zero, where the derivative of the
    # square root is infinite and would turn the masked gradient into NaN.
    distances = torch.sqrt(torch.where(keep, distance_squared, 1.0))

    return distances, keep


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
    nbl: NeighborList | None = None,
    **kwargs: Any,
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
    nbl : NeighborList | None, optional
        Neighbour list built at least at `cutoff`, to enumerate the triples
        from instead of the dense ``O(nat**3)`` evaluation; see
        :func:`tad_dftd3.damping.dispersion_atm`, whose `max_triples` and
        `checkpoint` can be passed as further keyword arguments. Unlike the
        two-body term, the memory of the sparse ATM term grows steeply with
        the cutoff, so it pays off for a `cutoff` smaller than the default
        or for large systems.

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
        nbl=nbl,
        **kwargs,
    )
