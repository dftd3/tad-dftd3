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
Axilrod-Teller-Muto (ATM) dispersion term
=========================================

This module provides the dispersion energy evaluation for the three-body
Axilrod-Teller-Muto dispersion term.

.. math::

    E_\text{disp}^{(3), \text{ATM}} &=
    \sum_\text{ABC} E^{\text{ABC}} f_\text{damp}\left(\overline{R}_\text{ABC}\right) \\
    E^{\text{ABC}} &=
    \dfrac{C^{\text{ABC}}_9
    \left(3 \cos\theta_\text{A} \cos\theta_\text{B} \cos\theta_\text{C} + 1 \right)}
    {\left(r_\text{AB} r_\text{BC} r_\text{AC} \right)^3} \\
    f_\text{damp} &=
    \dfrac{1}{1+ 6 \left(\overline{R}_\text{ABC}\right)^{-16}}
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.convert import any_to_tensor
from tad_mctc.io.structure import Structure
from tad_mctc.neighbor.list import NeighborList
from tad_mctc.neighbor.triples import TripleChunk, triples_from_neighborlist
from tad_mctc.typing import DD, TableFunction, Tensor
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from .. import defaults
from .._checks import require_molecule, takes_structure
from ..cutoff import smooth_cutoff
from ..data.table import element_table, reject_renamed_tables

__all__ = ["dispersion_atm"]


@reject_renamed_tables
@takes_structure
def dispersion_atm(
    structure: Structure,
    c6: Tensor,
    *,
    rvdw_table: Tensor | TableFunction | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    width: float = defaults.D3_DISP3_WIDTH,
    s9: Tensor | float | None = None,
    rs9: Tensor | float | None = None,
    alp: Tensor | float | None = None,
    nbl: NeighborList | None = None,
    max_triples: int = 2_000_000,
    checkpoint: bool = False,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    structure : Structure
        The system, a molecule: atomic numbers and Cartesian coordinates,
        single or batched (see :func:`tad_dftd3.disp.dftd3`).
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, indexed by atomic numbers, of
        shape ``(104, 104)``, or ``None`` for
        :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    cutoff : float, optional
        Real-space cutoff, in Bohr. Defaults to
        :data:`tad_dftd3.defaults.D3_DISP3_CUTOFF`.
    width : float, optional
        Width of the smooth cutoff, in Bohr: a triple is scaled by the switch
        of each of its three distances, see :mod:`tad_dftd3.cutoff`. Defaults
        to zero, a hard cutoff.
    s9 : Tensor | float, optional
        Scaling for dispersion coefficients. Defaults to `1.0`.
    rs9 : Tensor | float, optional
        Scaling for van-der-Waals radii in damping function. Defaults to `4.0/3.0`.
    alp : Tensor | float, optional
        Exponent of zero damping function. Defaults to `14.0`.
    nbl : NeighborList | None, optional
        A pre-built neighbour list of `structure`, built at least at
        `cutoff`. ``None`` (default) runs the dense, all-triples evaluation,
        with ``O(nat**3)`` memory whatever the cutoff. With a list, the
        triples within the cutoff are enumerated from it by
        :func:`tad_mctc.neighbor.triples.triples_from_neighborlist`, so
        memory follows the cutoff instead (it grows as ``cutoff**6`` per
        atom, so a smaller `cutoff` than the default is needed for large
        systems). A batch is a flat system numbered ``b * nat + i``.
    max_triples : int, optional
        Only with `nbl`. Upper bound on the triples held in memory at once
        in the forward pass. Defaults to ``2_000_000``.
    checkpoint : bool, optional
        Only with `nbl`. Recompute each chunk in the backward pass instead
        of keeping its intermediates. Any derivative order, but not
        ``vmap``. Defaults to ``False``.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.

    Raises
    ------
    ValueError
        If `structure` is a periodic cell, for which the ATM term has no
        evaluation.
    TypeError
        If `structure` is not a ``Structure``.
    """
    require_molecule(structure, "three-body (ATM) term")

    numbers, positions = structure.numbers, structure.positions
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    s9 = any_to_tensor(defaults.S9 if s9 is None else s9, **dd)
    rs9 = any_to_tensor(defaults.RS9 if rs9 is None else rs9, **dd)
    alp = any_to_tensor(defaults.ALP if alp is None else alp, **dd)

    table = element_table(rvdw_table, "rvdw_table", positions)
    srvdw = rs9 * table[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]

    if nbl is not None:
        nbl.check_compatible(structure, cutoff)
        return _sparse_dispersion_atm(
            structure,
            c6,
            srvdw,
            cutoff,
            width,
            s9,
            alp,
            nbl,
            max_triples=max_triples,
            checkpoint=checkpoint,
        )

    cutoff2 = cutoff * cutoff

    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_self=True)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    one = torch.tensor(1.0, **dd)

    # C9_ABC = s9 * sqrt(|C6_AB * C6_AC * C6_BC|)
    c9 = s9 * storch.safe_sqrt(
        torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3))
    )

    r0ij = srvdw.unsqueeze(-1)
    r0ik = srvdw.unsqueeze(-2)
    r0jk = srvdw.unsqueeze(-3)
    r0 = r0ij * r0ik * r0jk

    # actually faster than other alternatives
    # very slow: (pos.unsqueeze(-2) - pos.unsqueeze(-3)).pow(2).sum(-1)
    distances = torch.pow(
        torch.where(
            mask_pairs,
            storch.cdist(positions, positions, p=2),
            eps,
        ),
        2.0,
    )

    r2ij = distances.unsqueeze(-1)
    r2ik = distances.unsqueeze(-2)
    r2jk = distances.unsqueeze(-3)
    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    # add epsilon to avoid zero division later
    r3 = torch.where(mask_triples, r1 * r2, eps)
    r5 = torch.where(mask_triples, r2 * r3, eps)

    # dividing by tiny numbers leads to huge numbers, which result in NaN's
    # upon exponentiation in the subsequent step
    mask = real_triples(numbers, mask_self=True)
    base = r0 / torch.where(mask_triples, r1, one)

    # to fix the previous mask, we mask again (not strictly necessary because
    # `ang` is also masked and we later multiply with `ang`)
    fdamp = torch.where(
        mask_triples,
        1.0 / (1.0 + 6.0 * base ** ((alp + 2.0) / 3.0)),
        zero,
    )

    s = torch.where(
        mask,
        (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik),
        zero,
    )

    ang = torch.where(
        mask_triples
        * (r2ij <= cutoff2)
        * (r2ik <= cutoff2)
        * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        torch.tensor(0.0, **dd),
    )

    # smooth cutoff of each of the three distances
    if width > 0.0:
        ang = (
            ang
            * smooth_cutoff(torch.sqrt(r2ij), cutoff, width)
            * smooth_cutoff(torch.sqrt(r2ik), cutoff, width)
            * smooth_cutoff(torch.sqrt(r2jk), cutoff, width)
        )

    energy = ang * fdamp * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0


def _pair_table(
    table: Tensor, atom_a: Tensor, atom_b: Tensor, nat: int
) -> Tensor:
    """
    Entries ``table[..., a, b]`` of a pair table ``(..., nat, nat)`` for flat
    atom indices ``system * nat + i``: row ``system * nat + a`` of the
    reshaped table is ``table[system, a, :]``.
    """
    return table.reshape(-1, nat)[atom_a, atom_b % nat]


def _atm_chunk_energy(
    positions: Tensor,
    c6: Tensor,
    srvdw: Tensor,
    cutoff: float,
    width: float,
    s9: Tensor,
    alp: Tensor,
    chunk: TripleChunk,
    nat: int,
) -> Tensor:
    """
    A third of the ATM energy of every triple of one chunk, which the caller
    adds to each of the three atoms.

    Fixed-index-set tensor algebra, so it differentiates to any order.
    `positions` are the flattened ``(B * nat, 3)`` positions.
    """
    idx_i, idx_j, idx_k = chunk.idx_i, chunk.idx_j, chunk.idx_k

    r0 = (
        _pair_table(srvdw, idx_i, idx_j, nat)
        * _pair_table(srvdw, idx_i, idx_k, nat)
        * _pair_table(srvdw, idx_j, idx_k, nat)
    )
    c9 = s9 * storch.safe_sqrt(
        torch.abs(
            _pair_table(c6, idx_i, idx_j, nat)
            * _pair_table(c6, idx_i, idx_k, nat)
            * _pair_table(c6, idx_j, idx_k, nat)
        )
    )

    centre = positions.index_select(0, idx_j)
    v_ji = positions.index_select(0, idx_i) - centre
    v_jk = positions.index_select(0, idx_k) - centre
    v_ik = v_jk - v_ji

    r2ij_raw = v_ji.pow(2).sum(-1)
    r2ik_raw = v_ik.pow(2).sum(-1)
    r2jk_raw = v_jk.pow(2).sum(-1)

    # The triples are pruned to the cutoff already; masked again before any
    # square root, so that this never relies on the builder alone. A triple
    # is kept as a whole or dropped, like s-dftd3.
    cutoff2 = cutoff * cutoff
    mask = (r2ij_raw <= cutoff2) & (r2ik_raw <= cutoff2) & (r2jk_raw <= cutoff2)

    one = torch.ones_like(r2ij_raw)
    r2ij = torch.where(mask, r2ij_raw, one)
    r2ik = torch.where(mask, r2ik_raw, one)
    r2jk = torch.where(mask, r2jk_raw, one)

    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    base = r0 / r1
    fdamp = 1.0 / (1.0 + 6.0 * base ** ((alp + 2.0) / 3.0))

    s = (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik)
    ang = torch.where(mask, 0.375 * s / r5 + 1.0 / r3, torch.zeros_like(r5))

    # smooth cutoff of each of the three distances
    if width > 0.0:
        ang = (
            ang
            * smooth_cutoff(torch.sqrt(r2ij), cutoff, width)
            * smooth_cutoff(torch.sqrt(r2ik), cutoff, width)
            * smooth_cutoff(torch.sqrt(r2jk), cutoff, width)
        )

    # Each triple is listed once; the dense `/ 6.0` over its six
    # permutations is the equal share of its three atoms.
    return ang * fdamp * c9 / 3.0


def _sparse_dispersion_atm(
    structure: Structure,
    c6: Tensor,
    srvdw: Tensor,
    cutoff: float,
    width: float,
    s9: Tensor,
    alp: Tensor,
    nbl: NeighborList,
    *,
    max_triples: int,
    checkpoint: bool,
) -> Tensor:
    """
    ATM energy from the triples of a neighbour list, consumed chunk by chunk
    (at most `max_triples` at a time) into a per-atom accumulator.

    This bounds the forward working set only. Under autograd each chunk's
    index tensors stay alive until the backward pass, and so do its
    intermediates unless `checkpoint` recomputes them. The number of triples
    is data dependent, so unlike the sparse pair sums this path is not
    claimed to survive ``vmap`` or ``torch.compile(fullgraph=True)``.
    """
    numbers, positions = structure.numbers, structure.positions
    nat = positions.shape[-2]
    flat_positions = positions.reshape(-1, 3)

    counts = flat_positions.new_zeros(flat_positions.shape[0])
    for chunk in triples_from_neighborlist(
        nbl, structure, cutoff, chunk_size=max_triples
    ):
        if chunk.idx_i.shape[0] == 0:
            continue

        args = (flat_positions, c6, srvdw, cutoff, width, s9, alp, chunk, nat)
        if checkpoint:
            share = _torch_checkpoint(
                _atm_chunk_energy, *args, use_reentrant=False
            )
        else:
            share = _atm_chunk_energy(*args)

        # Three scatters: one over a concatenation of the indices would keep
        # another `3n` int64 tensor alive for backward.
        counts = counts.index_add(0, chunk.idx_i, share)
        counts = counts.index_add(0, chunk.idx_j, share)
        counts = counts.index_add(0, chunk.idx_k, share)

    return counts.reshape(numbers.shape)
