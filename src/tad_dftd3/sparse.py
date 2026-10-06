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
Neighbour-list evaluation
=========================

The two- and three-body terms summed over a pre-built
:class:`~tad_mctc.neighbor.list.NeighborList` instead of all pairs (and
periodic images), for molecules, batches and cells alike. They are selected
by passing a list to :func:`tad_dftd3.dftd3` or to the functions of
:mod:`tad_dftd3.disp`, and must agree with the dense evaluation.

The lists follow the conventions of
:func:`tad_mctc.ncoord.common.sum_over_neighborlist`:

- A batch is one flat system: atom ``i`` of system ``b`` is ``b * nat + i``,
  and entry ``(a, b)`` of a pair table ``(..., nat, nat)`` is
  ``table.reshape(-1, nat)[a, b % nat]``.
- Padded slots point at a phantom atom appended to the positions.
- The positions are the ones given, not folded into the cell: the shift of
  each entry includes the fold.

The three-body term runs over the triangles of its list, a
:class:`~tad_mctc.neighbor.triples.TripleList` (re-exported here as
:class:`TripleList`): the distance, C6 and radii of each pair of the list are
formed once, and each triple only looks up those of its three sides. Given
a list, the term builds its triangles when it is called, which is data
dependent. Built beforehand and passed instead of the list, they are reused
for as long as the list is (e.g. over the steps of a molecular dynamics),
and the term is fixed-shape tensor algebra, which survives
``torch.compile(fullgraph=True)`` and ``vmap``.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import sum_over_neighborlist
from tad_mctc.neighbor import pair_distance_squared, split_lattice
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.neighbor.triples import TripleList
from tad_mctc.typing import TableFunction, Tensor
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from . import defaults
from .cutoff import smooth_cutoff
from .damping import (
    DampingParam,
    PairData,
    ThreeBodyDamping,
    TwoBodyDamping,
    ZeroThreeBodyD3,
    atm_term,
)
from .data.table import element_table
from .model.c6 import AtomicC6

__all__ = ["TripleList", "dispersion2_sparse", "dispersion_atm_sparse"]


def dispersion2_sparse(
    structure: Structure,
    param: DampingParam,
    c6: Tensor | AtomicC6,
    nbl: NeighborList,
    *,
    damping: TwoBodyDamping,
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: float = defaults.D3_DISP2_CUTOFF,
    width: float = defaults.D3_DISP2_WIDTH,
    checkpoint: bool = False,
) -> Tensor:
    """
    Two-body dispersion energy summed over a neighbour list. This takes
    ``O(n_pairs)`` memory, and survives ``vmap`` and
    ``torch.compile(fullgraph=True)``: it is fixed-shape tensor algebra
    (unless `checkpoint` is set).

    Each entry of the list stands for the pair in both directions and the
    walk adds its contribution to both atoms, also for an atom with its own
    image, so every entry carries the ``-0.5`` of the dense sum.

    Parameters
    ----------
    structure : Structure
        The system, a molecule or a periodic cell, see
        :func:`tad_dftd3.disp.dftd3`.
    param : DampingParam
        DFT-D3 damping parameters.
    c6 : Tensor | AtomicC6
        Atomic C6 dispersion coefficients, a matrix or factored. Factored
        (:class:`~tad_dftd3.model.AtomicC6`), nothing of size ``nat**2`` is
        built, also not for the gradient.
    nbl : NeighborList
        Neighbour list of `structure`, built at least at `cutoff`. A list
        built at a larger cutoff, or with a skin, gives the same energy.
    damping : TwoBodyDamping
        The damping.
    rvdw_table, r4r2_table : Tensor | TableFunction | None, optional
        Element tables, see :func:`tad_dftd3.disp.dispersion2`.
    cutoff : float, optional
        Real-space cutoff, in Bohr.
    width : float, optional
        Width of the smooth cutoff, in Bohr. Zero is a hard cutoff.
    checkpoint : bool, optional
        Recompute each chunk of the list in the backward pass instead of
        keeping its intermediates (mode ``"recompute"`` of
        :func:`tad_mctc.ncoord.common.sum_over_neighborlist`). Any derivative
        order, but not ``vmap`` or ``torch.compile``. Defaults to ``False``.

    Returns
    -------
    Tensor
        Atom-resolved two-body dispersion energy.

    Raises
    ------
    ValueError
        If `nbl` does not match `structure` or does not cover `cutoff`.
    """
    nbl.check_compatible(structure, cutoff)

    numbers, positions = structure.numbers, structure.positions
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)

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

    numbers_flat = numbers.reshape(-1)

    # Ones for padding atoms, see `tad_dftd3.disp._dispersion2_dense`.
    r4r2_flat = _nonzero(r4r2[numbers_flat])

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
        # the square root: a padded slot is `sqrt(0)` otherwise, and its
        # derivative is infinite.
        mask = mask & (distance_squared <= cutoff * cutoff)
        distances = torch.sqrt(torch.where(mask, distance_squared, 1.0))

        # The tables know nothing about the phantom atom, so padded slots
        # are clamped to a real index, and masked below.
        real_i = idx_i.clamp(max=total_atoms - 1)
        real_j = idx_j.clamp(max=total_atoms - 1)
        c6_pair = _pair_c6(c6, real_i, real_j, nat)

        z_i, z_j = numbers_flat[real_i], numbers_flat[real_j]
        qq = 3 * r4r2_flat[real_i] * r4r2_flat[real_j]
        pairs = PairData(
            distances,
            qq,
            c6_pair,
            _nonzero(rvdw[z_i, z_j]),
            _nonzero(z_i + z_j).to(distances.dtype),
            torch.sqrt(qq),
        )

        kernel = damping(pairs, param)
        if width > 0.0:  # a static choice, the hard cutoff needs no switch
            kernel = smooth_cutoff(distances, cutoff, width) * kernel
        contribution = -0.5 * c6_pair * kernel
        contribution = torch.where(
            mask, contribution, torch.zeros_like(contribution)
        )
        return contribution, contribution

    energy = sum_over_neighborlist(
        nbl,
        pair_contributions,
        flat_positions,
        mode="recompute" if checkpoint else "graph",
    )
    return energy.reshape(numbers.shape)


def _nonzero(x: Tensor) -> Tensor:
    """`x`, with ones in place of zeros."""
    return torch.where(x != 0, x, torch.ones_like(x))


def _pair_c6(
    c6: Tensor | AtomicC6, idx_i: Tensor, idx_j: Tensor, nat: int
) -> Tensor:
    """
    The C6 coefficients of the pairs ``(idx_i, idx_j)`` of a list, in its flat
    numbering of the atoms of a batch.

    From an :class:`~tad_dftd3.model.AtomicC6`, as :meth:`D3Model
    <tad_dftd3.disp.D3Model>` passes it, they are computed per pair, and
    nothing of size ``nat**2`` is built. From the ``(..., nat, nat)``
    matrix, they are looked up, which is cheap in the forward pass but,
    when the matrix needs a gradient, creates one of its full size per chunk
    of pairs in the backward pass: quadratic in memory, cubic in time.

    Parameters
    ----------
    c6 : Tensor | AtomicC6
        Atomic C6 dispersion coefficients, as a matrix or factored.
    idx_i, idx_j : Tensor
        Flat indices of the two atoms of each pair, ``(n,)``.
    nat : int
        Atoms per system.

    Returns
    -------
    Tensor
        C6 coefficients of the pairs, ``(n,)``.
    """
    if isinstance(c6, AtomicC6):
        return c6.pair(idx_i, idx_j)
    return c6.reshape(-1, nat)[idx_i, idx_j % nat]


def dispersion_atm_sparse(
    structure: Structure,
    c6: Tensor | AtomicC6,
    param: DampingParam,
    nbl: NeighborList | TripleList,
    *,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    width: float = defaults.D3_DISP3_WIDTH,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term over the triangles of a neighbour
    list (see :class:`~tad_mctc.neighbor.triples.TripleList`), for a
    molecule or a cell, in chunks of at most `max_triples` triples.

    The squared distance, C6 coefficient and radii of each pair of the list
    are formed once; each triple looks up those of its three sides and is
    dropped as a whole if one of them is beyond `cutoff`.

    Memory follows the triangles of the list, which grow as the sixth power
    of its ``cutoff + skin`` per atom, so a smaller `cutoff` than the
    default is needed for large systems. The chunking bounds the forward
    working set only: under autograd the intermediates of each chunk stay
    alive until the backward pass, unless `checkpoint` recomputes them.

    Parameters
    ----------
    structure : Structure
        The system, a molecule or a periodic cell, see
        :func:`tad_dftd3.disp.dftd3`.
    c6 : Tensor | AtomicC6
        Atomic C6 dispersion coefficients, a matrix or factored. Factored
        (:class:`~tad_dftd3.model.AtomicC6`), nothing of size ``nat**2`` is
        built, also not for the gradient.
    param : DampingParam
        DFT-D3 damping parameters, read by `damping`.
    nbl : NeighborList | TripleList
        Neighbour list of `structure`, built at least at `cutoff`, whose
        triangles are built here, or its triangles, built beforehand. Only
        those are fixed-shape, for ``torch.compile(fullgraph=True)`` and
        ``vmap``.
    damping, rvdw_table, r4r2_table, cutoff, width
        See :func:`tad_dftd3.damping.dispersion_atm`.
    max_triples : int, optional
        Upper bound on the triples evaluated at once in the forward pass;
        the indices of all triangles of the list (six integers each) are
        held for the whole call. Defaults to ``2_000_000``.
    checkpoint : bool, optional
        Recompute each chunk in the backward pass instead of keeping its
        intermediates. Any derivative order, but not ``vmap``. Defaults to
        ``False``.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.

    Raises
    ------
    ValueError
        If the list does not match `structure` or does not cover `cutoff`.
    """
    if isinstance(nbl, TripleList):
        triples = nbl
        triples.check_compatible(structure, cutoff)
    else:
        nbl.check_compatible(structure, cutoff)
        triples = TripleList.from_neighborlist(nbl, chunk_size=max_triples)

    numbers, positions = structure.numbers, structure.positions
    rvdw = element_table(rvdw_table, "rvdw_table", positions)
    r4r2 = element_table(r4r2_table, "r4r2_table", positions)

    nat = positions.shape[-2]
    flat_positions = positions.reshape(-1, 3)
    total_atoms = flat_positions.shape[0]

    # Of every slot of the list, once. Padded slots point at a phantom atom,
    # which no triple refers to: its position is a constant, and its tables
    # are those of the last atom.
    pairs = triples.pairs
    padded_positions = torch.cat(
        [flat_positions, flat_positions.new_zeros(1, 3)]
    )
    shared_lattice, system_lattices = split_lattice(structure.lattice)
    if system_lattices is not None:
        system_lattices = torch.cat(
            [system_lattices, system_lattices.new_zeros(1, 3, 3)]
        )
    r2 = pair_distance_squared(
        pairs.idx_i,
        pairs.idx_j,
        pairs.shift,
        padded_positions,
        shared_lattice=shared_lattice,
        system_lattices=system_lattices,
        atoms_per_system=nat,
    )

    real_i = pairs.idx_i.clamp(max=total_atoms - 1)
    real_j = pairs.idx_j.clamp(max=total_atoms - 1)
    c6_pairs = _pair_c6(c6, real_i, real_j, nat)

    # the radii of `tad_dftd3.damping.pair_radii`, per slot
    numbers_flat = numbers.reshape(-1)
    r4r2_flat = _nonzero(r4r2[numbers_flat])
    rdamp = torch.sqrt(3.0 * r4r2_flat[real_i] * r4r2_flat[real_j])
    rvdw_pairs = _nonzero(rvdw[numbers_flat[real_i], numbers_flat[real_j]])
    radii = torch.stack((rdamp, rvdw_pairs), dim=-1)

    total = triples.idx_i.shape[0]
    size = max(max_triples, 1)
    if torch.compiler.is_compiling():
        # The compiler fuses the kernel; a chunk loop would only unroll.
        size = max(total, 1)

    # At least one chunk, empty without triples, so that the energy is still
    # part of the graph of the positions (with a zero gradient).
    energy = flat_positions.new_zeros(total_atoms)
    for start in range(0, max(total, 1), size):
        rows = slice(start, start + size)
        args = (
            r2,
            c6_pairs,
            radii,
            triples.side_ij[rows],
            triples.side_jk[rows],
            triples.side_ik[rows],
            damping,
            param,
            cutoff,
            width,
        )
        if checkpoint:
            share = _torch_checkpoint(
                _triple_energy, *args, use_reentrant=False
            )
        else:
            share = _triple_energy(*args)

        # Three scatters: one over a concatenation of the indices would keep
        # another `3n` int64 tensor alive for backward.
        energy = energy.index_add(0, triples.idx_i[rows], share)
        energy = energy.index_add(0, triples.idx_j[rows], share)
        energy = energy.index_add(0, triples.idx_k[rows], share)

    return energy.reshape(numbers.shape)


def _triple_energy(
    r2: Tensor,
    c6: Tensor,
    radii: Tensor,
    side_ij: Tensor,
    side_jk: Tensor,
    side_ik: Tensor,
    damping: ThreeBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
) -> Tensor:
    """
    A third of the ATM energy of each triple, which the caller adds to each
    of its three atoms, from the squared distance `r2`, C6 coefficient `c6`
    and `radii` of every slot of the list, and the slots of the sides.

    Fixed-index-set tensor algebra, so it differentiates to any order.
    """

    def of(table: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            table.index_select(0, side_ij),
            table.index_select(0, side_ik),
            table.index_select(0, side_jk),
        )

    r2ij, r2ik, r2jk = of(r2)
    term = atm_term(r2ij, r2ik, r2jk, of(radii), damping, param, cutoff, width)
    c6ij, c6ik, c6jk = of(c6)
    c9 = storch.safe_sqrt(torch.abs(c6ij * c6ik * c6jk))

    # Each triple is listed once; the dense `/ 6.0` over its six
    # permutations is the equal share of its three atoms.
    return term * c9 / 3.0
