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

How the pairs (and triples) of a term are enumerated is chosen by what is
passed as its `pairs`:

- ``None``: all pairs of a molecule, or of a cell with every periodic image
  within the cutoff (the image shifts are built here, eagerly);
- a :class:`~tad_mctc.neighbor.images.PeriodicShifts`: the same for a cell,
  with pre-built shifts, as needed under ``torch.func`` and
  ``torch.compile``;
- a :class:`~tad_mctc.neighbor.list.NeighborList`: the sparse evaluation of
  :mod:`tad_dftd3.sparse`;
- for the three-body term also a :class:`~tad_dftd3.sparse.TripleList`,
  the triangles of a list built beforehand, to reuse over many calls (e.g.
  the steps of a molecular dynamics) and as needed under
  ``torch.compile(fullgraph=True)`` and ``vmap``.

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
>>> ref = d3.reference.Reference.load(dtype=torch.double)
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

from collections.abc import Mapping
from typing import Any, TypeAlias

import torch
from tad_mctc import Structure, storch
from tad_mctc.batch import real_pairs
from tad_mctc.ncoord.common import _periodic_images
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.neighbor.list import NeighborList, build_neighborlists
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import DD, CountingFunction, TableFunction, Tensor

from . import model, ncoord
from .cutoff import Cutoff, smooth_cutoff
from .damping import (
    Damping,
    DampingParam,
    PairData,
    ThreeBodyDamping,
    TwoBodyDamping,
    ZeroThreeBodyD3,
    as_damping_param,
    damping_from_name,
    dispersion_atm,
    dispersion_atm_periodic,
)
from .data.table import element_table
from .model.c6 import AtomicC6
from .model.weights import WeightingFunction
from .reference import Reference, _default_reference
from .sparse import TripleList, dispersion2_sparse, dispersion_atm_sparse

__all__ = [
    "D3Model",
    "Pairs",
    "default_damping",
    "dftd3",
    "dispersion",
    "dispersion2",
    "dispersion3",
]


Pairs: TypeAlias = PeriodicShifts | NeighborList | None
"""How the pairs of a term are enumerated, see the module docstring."""

Param: TypeAlias = DampingParam | Mapping[str, Any]
"""Damping parameters, or a dictionary of them, see :func:`dftd3`."""


class D3Model(Node):
    """
    A DFT-D3 model as a value, like :data:`tad_mctc.ncoord.cn_d3` for the
    coordination number and like s-dftd3's ``DispersionModel``, except that
    it does not hold a structure: it is the *configuration* of the model,
    and is called on a structure (single or batch).

    ``D3Model()(structure, param)`` is :func:`dftd3`, which builds a model
    from its keyword arguments. Another variant is obtained with
    :meth:`replace`, e.g. ``model.replace(cutoff=Cutoff(disp2=50.0))``.

    A frozen :class:`~tad_mctc.tree.Node`, compared and hashed by identity. A
    tensor in one of the tables is a pytree leaf, so it can be differentiated
    or batched with ``torch.func``, and moved with :meth:`to`; a table
    function and the settings are static.

    Parameters
    ----------
    rcov_table : Tensor | TableFunction | None, optional
        Covalent radii per element, of shape ``(119,)``. ``None`` is
        :func:`tad_mctc.data.radii.COV_D3`.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
        ``None`` is :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
        ``None`` is :func:`tad_dftd3.data.R4R2`.
    ref : Reference | None, optional
        Reference C6 coefficients. ``None`` is the default reference on the
        device and dtype of the structure.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model.
    counting_function : CountingFunction, optional
        Counting function of the coordination number.
    weighting_function : WeightingFunction, optional
        Logarithm of the weight of a reference system, see
        :data:`tad_dftd3.model.WeightingFunction`.
    """

    rcov_table: Tensor | TableFunction | None = child(default=None)
    rvdw_table: Tensor | TableFunction | None = child(default=None)
    r4r2_table: Tensor | TableFunction | None = child(default=None)
    ref: Reference | None = child(default=None)
    cutoff: Cutoff = context(default_factory=Cutoff)
    counting_function: CountingFunction = context(default=ncoord.exp_count)
    weighting_function: WeightingFunction = context(
        default=model.gaussian_log_weight
    )

    def c6(
        self,
        structure: Structure,
        pairs: Pairs = None,
        *,
        checkpoint: bool = False,
    ) -> Tensor:
        """
        Atomic C6 dispersion coefficients of `structure`, of shape
        ``(..., nat, nat)``: the dense form of :meth:`factored_c6`, whose
        arguments these are.

        Returns
        -------
        Tensor
            C6 coefficients for all pairs of atoms.
        """
        return self.factored_c6(structure, pairs, checkpoint=checkpoint).dense()

    def factored_c6(
        self,
        structure: Structure,
        pairs: Pairs = None,
        *,
        checkpoint: bool = False,
    ) -> AtomicC6:
        """
        Atomic C6 dispersion coefficients of `structure`, factored per atom
        (see :class:`~tad_dftd3.model.AtomicC6`), as :meth:`__call__` passes
        them to the dispersion terms.

        Parameters
        ----------
        structure : Structure
            The system, a molecule or a periodic cell, see :func:`dftd3`.
        pairs : PeriodicShifts | NeighborList | None, optional
            How the pairs of the coordination number are enumerated, see
            :class:`tad_mctc.ncoord.common.CNModel`.
        checkpoint : bool, optional
            For a neighbour list, recompute each chunk of pairs of the
            coordination number in the backward pass (mode ``"recompute"``
            of :class:`~tad_mctc.ncoord.common.CNModel`). Defaults to
            ``False``.

        Returns
        -------
        AtomicC6
            C6 coefficients for all pairs of atoms.
        """
        ref = self.ref
        if ref is None:
            ref = _default_reference(structure.positions)

        cn = self.coordination_number(structure, pairs, checkpoint=checkpoint)
        weights = model.weight_references(
            structure.numbers, cn, ref, self.weighting_function
        )
        return model.factored_c6(structure.numbers, weights, ref)

    def coordination_number(
        self,
        structure: Structure,
        pairs: Pairs = None,
        *,
        checkpoint: bool = False,
    ) -> Tensor:
        """
        D3 coordination number of `structure`, as :meth:`factored_c6` uses
        it, whose arguments these are.

        Returns
        -------
        Tensor
            Coordination number of each atom, of the shape of
            ``structure.numbers``.
        """
        cn_model = ncoord.cn_d3.replace(
            count=self.counting_function,
            cutoff=self.cutoff.cn,
            rcov=element_table(
                self.rcov_table, "rcov_table", structure.positions
            ),
        )
        # `CNModel` takes the mode for a neighbour list only.
        recompute = checkpoint and isinstance(pairs, NeighborList)
        return cn_model(
            structure, pairs, mode="recompute" if recompute else "graph"
        )

    def __call__(
        self,
        structure: Structure,
        param: Param,
        *,
        damping: Damping | None = None,
        shifts: PeriodicShifts | None = None,
        nbl_cn: NeighborList | None = None,
        nbl_disp2: NeighborList | None = None,
        nbl_disp3: NeighborList | TripleList | None = None,
        sparse: bool = False,
        max_triples: int = 2_000_000,
        checkpoint: bool = False,
    ) -> Tensor:
        """
        Atom-resolved DFT-D3 dispersion energy, see :func:`dftd3`, whose
        arguments these are.

        Returns
        -------
        Tensor
            Energy of each atom, of the shape of ``structure.numbers``.
        """
        param = as_damping_param(param)
        if damping is None:
            damping = default_damping(param)

        if sparse:
            nbl_cn, nbl_disp2, nbl_disp3 = self._missing_lists(
                structure,
                nbl_cn,
                nbl_disp2,
                nbl_disp3,
                three_body=damping.three is not None,
            )

        c6 = self.factored_c6(
            structure,
            shifts if nbl_cn is None else nbl_cn,
            checkpoint=checkpoint,
        )
        return dispersion(
            structure,
            param,
            c6,
            damping=damping,
            shifts=shifts,
            nbl_disp2=nbl_disp2,
            nbl_disp3=nbl_disp3,
            rvdw_table=self.rvdw_table,
            r4r2_table=self.r4r2_table,
            cutoff=self.cutoff,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    def _missing_lists(
        self,
        structure: Structure,
        nbl_cn: NeighborList | None,
        nbl_disp2: NeighborList | None,
        nbl_disp3: NeighborList | TripleList | None,
        *,
        three_body: bool,
    ) -> tuple[NeighborList, NeighborList, NeighborList | TripleList | None]:
        """
        The lists of the coordination number, the two-body term and, with
        `three_body`, the three-body term, with the missing ones built in one
        shared search (at the largest of their cutoffs), eagerly: it is data
        dependent.
        """
        terms = [(nbl_cn, self.cutoff.cn), (nbl_disp2, self.cutoff.disp2)]
        build3 = three_body and nbl_disp3 is None

        missing = tuple(cutoff for nbl, cutoff in terms if nbl is None)
        if build3:
            missing += (self.cutoff.disp3,)
        built = iter(build_neighborlists(structure, missing) if missing else ())
        lists = [next(built) if nbl is None else nbl for nbl, _ in terms]

        return lists[0], lists[1], next(built) if build3 else nbl_disp3


def dftd3(
    structure: Structure,
    param: Param,
    *,
    damping: Damping | None = None,
    shifts: PeriodicShifts | None = None,
    nbl_cn: NeighborList | None = None,
    nbl_disp2: NeighborList | None = None,
    nbl_disp3: NeighborList | TripleList | None = None,
    sparse: bool = False,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
    ref: Reference | None = None,
    rcov_table: Tensor | TableFunction | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = model.gaussian_log_weight,
) -> Tensor:
    """
    Evaluate DFT-D3 dispersion energy for a batch of geometries.

    **Periodic cells.** If `structure` has a ``lattice``, it is a periodic
    cell along the axes of its ``periodic`` mask: the coordination number
    and the two-body energy sum over every periodic image within their
    cutoff, and the three-body term over the triples of an atom of the cell
    with two images, as in s-dftd3. Atoms need not lie inside the cell.

    Which images lie within a cutoff depends on the values of the lattice,
    so by default the image shifts are built eagerly on each call. Under
    ``torch.compile(fullgraph=True)``, ``vmap`` over cells, or
    ``jacrev``/``jacfwd`` with respect to the lattice, build them once
    beforehand, at the largest cutoff of the terms evaluated, and pass them
    as `shifts`::

        from tad_mctc.neighbor.images import build_periodic_shifts

        cutoff = Cutoff()
        shifts = build_periodic_shifts(
            structure.lattice,
            structure.periodic,
            max(cutoff.cn, cutoff.disp2, cutoff.disp3),
        )

    **Neighbour lists.** Instead of the dense, all-pairs sums, each term can
    run over a pre-built, padded neighbour list
    (:class:`tad_mctc.neighbor.list.NeighborList`, see
    :mod:`tad_dftd3.sparse`), which scales linearly with the number of
    atoms, for molecules, batches and cells alike. A list given for a term
    takes the place of `shifts` for that term. With ``sparse=True``, the
    lists that are not given are built eagerly on each call, sharing one
    search: of the coordination number, the two-body term and, if the
    damping has one, the three-body term. The triples of the last grow as
    ``cutoff**6`` per atom (see
    :func:`tad_dftd3.sparse.dispersion_atm_sparse`), so for a large system a
    ``cutoff.disp3`` below the default may be needed; the dense three-body
    term of a molecule needs memory cubic in the number of atoms, whatever
    the cutoff. To reuse the lists, e.g. over the steps of a molecular
    dynamics, build them once, each at (at least) its own cutoff, with a
    skin, and rebuild them when :meth:`~tad_mctc.neighbor.list.NeighborList.stale`
    says so: nothing checks that the atoms have not moved too far for the
    skin. The three-body term runs over the triangles of its list; built
    with the list (:class:`~tad_dftd3.sparse.TripleList`) and passed in its
    place, they are reused too::

        from tad_mctc.neighbor.list import build_neighborlists
        from tad_dftd3.sparse import TripleList

        cutoff = Cutoff()
        cutoffs = (cutoff.cn, cutoff.disp2, cutoff.disp3)

        def build(structure):
            nbl_cn, nbl_disp2, nbl_disp3 = build_neighborlists(
                structure, cutoffs, skin=1.0
            )
            triples = TripleList.from_neighborlist(nbl_disp3)
            return dict(nbl_cn=nbl_cn, nbl_disp2=nbl_disp2, nbl_disp3=triples)

        lists = build(structure)
        for step in trajectory:
            if lists["nbl_cn"].stale(structure):  # all were built together
                lists = build(structure)
            energy = dftd3(structure, param, cutoff=cutoff, **lists)
            ...

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
    param : DampingParam | Mapping[str, Tensor | float]
        DFT-D3 damping parameters. A dictionary is unpacked into a
        :class:`~tad_dftd3.damping.DampingParam`.
    damping : Damping | None, optional
        The damping of the two- and three-body term. Defaults to the one
        named by ``param.damping`` (rational damping if unset), with the
        zero-damped three-body term if `s9` is set and not the Python number
        zero. A tensor `s9` always gets the three-body term, even at zero:
        the choice must be static under ``torch.compile`` and ``vmap``, and
        the derivative with respect to `s9` is the three-body energy.
    shifts : PeriodicShifts | None, optional
        Periodic image shifts of a cell, at least at the cutoffs of the
        terms that do not have a neighbour list. Defaults to building them
        on each call.
    nbl_cn, nbl_disp2, nbl_disp3 : NeighborList | None, optional
        Neighbour lists of the coordination number, the two-body and the
        three-body term, each built at least at its cutoff. For the
        three-body term, also the triangles of its list
        (:meth:`TripleList.from_neighborlist
        <tad_dftd3.sparse.TripleList.from_neighborlist>`), built once and
        valid as long as the list: reused over the steps of a molecular
        dynamics, and needed under ``torch.compile(fullgraph=True)`` or
        ``vmap``.
    sparse : bool, optional
        Build the lists that are not given: of the coordination number, the
        two-body term and, if the damping has one, the three-body term.
        Defaults to ``False``.
    max_triples : int, optional
        For the three-body term of a cell or over a neighbour list, the
        upper bound on the triples evaluated at once in the forward pass.
        Over a list, the indices of all its triangles are held for the
        whole call. Defaults to ``2_000_000``.
    checkpoint : bool, optional
        Recompute each chunk of pairs (over a neighbour list) and of
        triples (of a cell or over a neighbour list) in the backward pass
        instead of keeping its intermediates, so that autograd holds the
        intermediates of one chunk at a time. Any derivative order, but not
        ``vmap`` or ``torch.compile``. Defaults to ``False``.
    ref : Reference | None, optional
        Reference C6 coefficients.
    rcov_table : Tensor | TableFunction | None, optional
        Covalent radii per element, of shape ``(119,)``. Defaults to
        :func:`tad_mctc.data.radii.COV_D3`.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``.
        Defaults to :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``.
        Defaults to :func:`tad_dftd3.data.R4R2`.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model.
    counting_function : CountingFunction, optional
        Calculates counting value in range 0 to 1 for each atom pair.
    weighting_function : WeightingFunction, optional
        Logarithm of the weight of a reference system, see
        :data:`tad_dftd3.model.WeightingFunction`.

    The element tables are indexed by atomic number (entry 0 is the dummy),
    not per-atom values. Gradients with respect to them come out per
    element, summed over all atoms and systems.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D3 dispersion energy for each geometry, of the
        shape of ``structure.numbers``.

    Raises
    ------
    ValueError
        If an element table does not have the shape of its default, a
        parameter the damping needs is not set, or `shifts` or a neighbour
        list does not match the structure or its cutoff.
    """
    return D3Model(
        rcov_table=rcov_table,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        ref=ref,
        cutoff=cutoff,
        counting_function=counting_function,
        weighting_function=weighting_function,
    )(
        structure,
        param,
        damping=damping,
        shifts=shifts,
        nbl_cn=nbl_cn,
        nbl_disp2=nbl_disp2,
        nbl_disp3=nbl_disp3,
        sparse=sparse,
        max_triples=max_triples,
        checkpoint=checkpoint,
    )


def default_damping(param: DampingParam) -> Damping:
    """
    The damping named by ``param.damping`` (rational damping if unset), with
    the zero damping of the three-body term if `s9` is set and not the
    Python number zero.

    This is a static choice: a Python number is a constant, also to
    ``torch.compile``, and a tensor `s9` always gets the three-body term.
    """
    s9 = param.s9
    three_body = s9 is not None and (isinstance(s9, Tensor) or s9 != 0.0)
    return damping_from_name(param.damping or "rational", three_body)


def dispersion(
    structure: Structure,
    param: Param,
    c6: Tensor | AtomicC6,
    *,
    damping: Damping | None = None,
    shifts: PeriodicShifts | None = None,
    nbl_disp2: NeighborList | None = None,
    nbl_disp3: NeighborList | TripleList | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Dispersion energy from given C6 coefficients: the two-body term, and the
    three-body term if `damping` has one.

    Parameters
    ----------
    structure : Structure
        The system, a molecule or a periodic cell, see :func:`dftd3`.
    param : DampingParam | Mapping[str, Tensor | float]
        DFT-D3 damping parameters.
    c6 : Tensor | AtomicC6
        Atomic C6 dispersion coefficients, as the ``(..., nat, nat)`` matrix
        or factored per atom (:class:`~tad_dftd3.model.AtomicC6`, as from
        :meth:`D3Model.factored_c6`). Over a neighbour list, the factored
        form takes no memory quadratic in the number of atoms.
    damping : Damping | None, optional
        The damping. Defaults to :func:`default_damping` of `param`.
    shifts, nbl_disp2, nbl_disp3
        How the pairs of each term are enumerated, see :func:`dftd3`.
    rvdw_table, r4r2_table : Tensor | TableFunction | None, optional
        Element tables, see :func:`dftd3`.
    cutoff : Cutoff, optional
        Real-space cutoffs, one per part of the model.
    max_triples, checkpoint
        See :func:`dftd3`.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D3 dispersion energy for each geometry.
    """
    param = as_damping_param(param)
    if damping is None:
        damping = default_damping(param)

    # A term without a neighbour list needs the C6 matrix: built once for
    # both, and only then.
    matrix = c6
    if isinstance(c6, AtomicC6) and (
        nbl_disp2 is None or (damping.three is not None and nbl_disp3 is None)
    ):
        matrix = c6.dense()

    energy = dispersion2(
        structure,
        param,
        matrix if nbl_disp2 is None else c6,
        pairs=shifts if nbl_disp2 is None else nbl_disp2,
        damping=damping.two,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cutoff,
        checkpoint=checkpoint,
    )

    if damping.three is not None:
        # Not added in place: under `vmap` over `s9`, only the three-body
        # term is batched, and the two-body energy cannot take its batch
        # dimension.
        energy = energy + dispersion3(
            structure,
            param,
            matrix if nbl_disp3 is None else c6,
            pairs=shifts if nbl_disp3 is None else nbl_disp3,
            damping=damping.three,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            cutoff=cutoff,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    return energy


def dispersion2(
    structure: Structure,
    param: Param,
    c6: Tensor | AtomicC6,
    *,
    pairs: Pairs = None,
    damping: TwoBodyDamping | None = None,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    checkpoint: bool = False,
) -> Tensor:
    """
    Two-body dispersion energy.

    Densely, every atom of a cell is paired with every periodic image of
    every atom within ``cutoff.disp2``, including the images of the atom
    itself, which takes ``O(nat**2 * n_shift)`` memory; over a neighbour
    list, see :func:`tad_dftd3.sparse.dispersion2_sparse`.

    Parameters
    ----------
    structure : Structure
        The system, a molecule or a periodic cell, see :func:`dftd3`.
    param : DampingParam | Mapping[str, Tensor | float]
        DFT-D3 damping parameters.
    c6 : Tensor | AtomicC6
        Atomic C6 dispersion coefficients, as the ``(..., nat, nat)`` matrix
        or factored per atom (:class:`~tad_dftd3.model.AtomicC6`, as from
        :meth:`D3Model.factored_c6`). Over a neighbour list, the factored
        form takes no memory quadratic in the number of atoms.
    pairs : PeriodicShifts | NeighborList | None, optional
        How the pairs are enumerated, see the module docstring. Built at
        least at ``cutoff.disp2``.
    damping : TwoBodyDamping | None, optional
        The damping. Defaults to the two-body part of
        :func:`default_damping` of `param`.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``, or
        ``None`` for :func:`tad_mctc.data.radii.VDW_PAIRWISE`. Used by the
        zero damping.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``, or
        ``None`` for :func:`tad_dftd3.data.R4R2`.
    cutoff : Cutoff, optional
        Real-space cutoffs, of which `disp2` and `width2` are used: the
        contribution of a pair is scaled down to zero over the last `width2`
        below `disp2`, see :mod:`tad_dftd3.cutoff`.
    checkpoint : bool, optional
        For a neighbour list, recompute each chunk of pairs in the backward
        pass instead of keeping its intermediates. Not under ``vmap`` or
        ``torch.compile``. Defaults to ``False``.

    Returns
    -------
    Tensor
        Atom-resolved two-body dispersion energy.

    Raises
    ------
    ValueError
        If `pairs` does not match `structure` or does not cover the cutoff.
    """
    param = as_damping_param(param)
    if damping is None:
        damping = default_damping(param).two

    if isinstance(pairs, NeighborList):
        return dispersion2_sparse(
            structure,
            param,
            c6,
            pairs,
            damping=damping,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            cutoff=cutoff.disp2,
            width=cutoff.width2,
            checkpoint=checkpoint,
        )

    return _dispersion2_dense(
        structure,
        param,
        _matrix(c6),
        pairs,
        damping=damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cutoff.disp2,
        width=cutoff.width2,
    )


def _dispersion2_dense(
    structure: Structure,
    param: DampingParam,
    c6: Tensor,
    shifts: PeriodicShifts | None,
    *,
    damping: TwoBodyDamping,
    rvdw_table: Tensor | TableFunction | None,
    r4r2_table: Tensor | TableFunction | None,
    cutoff: float,
    width: float,
) -> Tensor:
    """
    Two-body energy over all pairs of a molecule, ``(..., nat, nat)``, or of
    a cell with every periodic image, ``(..., nat, nat, n_shift)``.
    """
    if shifts is not None:
        # Also rejects shifts for a molecule, which would be ignored.
        shifts.check_compatible(structure, cutoff)

    numbers, positions = structure.numbers, structure.positions
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)

    # Padding atoms look up the dummy entry 0 of the tables, which is zero,
    # and a damping divides by `r4r2` (its gradient of `sqrt(qq)` would be
    # infinite), the radius and the atomic number. Masking only the output
    # does not help (0 * inf = NaN), so the inputs are replaced by ones too,
    # as are the distances, and masked below.
    r4r2_atom = _nonzero(r4r2[numbers])
    qq = 3 * r4r2_atom.unsqueeze(-1) * r4r2_atom.unsqueeze(-2)
    rvdw_pairs = _nonzero(rvdw[numbers.unsqueeze(-1), numbers.unsqueeze(-2)])
    znum = _nonzero(numbers.unsqueeze(-1) + numbers.unsqueeze(-2))
    znum = znum.to(positions.dtype)

    # the damping radius of D3 (and D4) for the dampings ported from dftd
    rdamp = torch.sqrt(qq)

    if structure.lattice is None:
        distances, keep = _molecular_distances(numbers, positions, cutoff)
        pairs = PairData(distances, qq, c6, rvdw_pairs, znum, rdamp)
    else:
        if shifts is None:
            # `Structure` fills in a mask whenever it has a lattice.
            assert structure.periodic is not None
            shifts = build_periodic_shifts(
                structure.lattice, structure.periodic, cutoff
            )
        distances, keep = _periodic_distances(structure, shifts, cutoff)

        # One entry per pair and image, all with the pair's values.
        pairs = PairData(
            distances,
            qq.unsqueeze(-1),
            c6.unsqueeze(-1),
            rvdw_pairs.unsqueeze(-1),
            znum.unsqueeze(-1),
            rdamp.unsqueeze(-1),
        )

    kernel = damping(pairs, param)
    if width > 0.0:  # a static choice, the hard cutoff needs no switch
        kernel = smooth_cutoff(distances, cutoff, width) * kernel
    kernel = torch.where(keep, kernel, torch.zeros_like(kernel))

    if structure.lattice is not None:
        # C6 is the same for every image of a pair, so the damped terms are
        # summed over the images first and multiplied per pair.
        kernel = kernel.sum(dim=-1)

    return -0.5 * torch.sum(c6 * kernel, dim=-1)


def _nonzero(x: Tensor) -> Tensor:
    """`x`, with ones in place of zeros."""
    return torch.where(x != 0, x, torch.ones_like(x))


def _matrix(c6: Tensor | AtomicC6) -> Tensor:
    """The ``(..., nat, nat)`` C6 matrix, which the dense terms need."""
    return c6.dense() if isinstance(c6, AtomicC6) else c6


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


def dispersion3(
    structure: Structure,
    param: Param,
    c6: Tensor | AtomicC6,
    *,
    pairs: Pairs | TripleList = None,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: Cutoff = Cutoff(),
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Three-body dispersion term, the Axilrod-Teller-Muto term. Evaluated by
    one of :func:`tad_dftd3.damping.dispersion_atm` (molecule),
    :func:`tad_dftd3.damping.dispersion_atm_periodic` (cell) or
    :func:`tad_dftd3.sparse.dispersion_atm_sparse` (neighbour list).

    Parameters
    ----------
    structure : Structure
        The system, a molecule or a periodic cell (see :func:`dftd3`).
    param : DampingParam | Mapping[str, Tensor | float]
        DFT-D3 damping parameters.
    c6 : Tensor | AtomicC6
        Atomic C6 dispersion coefficients, as the ``(..., nat, nat)`` matrix
        or factored per atom (:class:`~tad_dftd3.model.AtomicC6`, as from
        :meth:`D3Model.factored_c6`). Over a neighbour list, the factored
        form takes no memory quadratic in the number of atoms.
    pairs : PeriodicShifts | NeighborList | TripleList | None, optional
        How the triples are enumerated, see the module docstring. Built at
        least at ``cutoff.disp3``.
    damping : ThreeBodyDamping, optional
        The damping, called for every triple. Defaults to zero damping.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, of shape ``(104, 104)``, or
        ``None`` for :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``, or
        ``None`` for :func:`tad_dftd3.data.R4R2`. Both tables give the radii
        of the pairs, see :func:`tad_dftd3.damping.pair_radii`.
    cutoff : Cutoff, optional
        Real-space cutoffs, of which `disp3` and `width3` are used, see
        :func:`dispersion2`. A triple is scaled by the switch of each of its
        three distances.
    max_triples : int, optional
        For a cell or a neighbour list, the upper bound on the triples
        evaluated at once in the forward pass. Over a list, the indices of
        all its triangles are held for the whole call. Defaults to
        ``2_000_000``.
    checkpoint : bool, optional
        For a cell or a neighbour list, recompute each block of triples in
        the backward pass instead of keeping its intermediates. Defaults to
        ``False``.

    Returns
    -------
    Tensor
        Atom-resolved three-body dispersion energy.

    Raises
    ------
    ValueError
        If `pairs` does not match `structure` or does not cover the cutoff.
    """
    param = as_damping_param(param)
    cut, width = cutoff.disp3, cutoff.width3

    if isinstance(pairs, (NeighborList, TripleList)):
        return dispersion_atm_sparse(
            structure,
            c6,
            param,
            pairs,
            damping=damping,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            cutoff=cut,
            width=width,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    # Shifts for a molecule are rejected there.
    if structure.lattice is not None or pairs is not None:
        return dispersion_atm_periodic(
            structure,
            _matrix(c6),
            param,
            pairs,
            damping=damping,
            rvdw_table=rvdw_table,
            r4r2_table=r4r2_table,
            cutoff=cut,
            width=width,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    return dispersion_atm(
        structure,
        _matrix(c6),
        param,
        damping=damping,
        rvdw_table=rvdw_table,
        r4r2_table=r4r2_table,
        cutoff=cut,
        width=width,
    )
