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

This module provides the dense evaluation of the three-body
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

The functions here evaluate the geometry of the term: the angular factor
and the cutoff of every triple. The damping, with the scaling `s9`, is a
:class:`~tad_dftd3.damping.ThreeBodyDamping` they call for every triple with
its :class:`~tad_dftd3.damping.TripleData`, as dftd calls ``get_3b_damp``;
D3 gives it the van-der-Waals radii of the pairs. There is one function per
way of enumerating the triples, without any dispatch between them (that is
done by :func:`tad_dftd3.disp.dispersion3`):

- :func:`dispersion_atm`: all triples of a molecule, ``O(nat**3)`` memory.
- :func:`dispersion_atm_periodic`: a cell, every atom of the cell with all
  pairs of periodic images within the cutoff, as s-dftd3.
- :func:`tad_dftd3.sparse.dispersion_atm_sparse`: the triples of a neighbour
  list, for a molecule or a cell.

The latter two share the energy of a triple, :func:`atm_term`.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.io.structure import Structure
from tad_mctc.ncoord.common import _periodic_images
from tad_mctc.neighbor.images import PeriodicShifts, build_periodic_shifts
from tad_mctc.typing import DD, TableFunction, Tensor
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from .. import defaults
from ..cutoff import smooth_cutoff
from ..data.table import element_table
from .base import ThreeBodyDamping, TripleData
from .param import DampingParam
from .zero import ZeroThreeBodyD3

__all__ = [
    "atm_term",
    "dispersion_atm",
    "dispersion_atm_periodic",
    "pair_radii",
]


def dispersion_atm(
    structure: Structure,
    c6: Tensor,
    param: DampingParam,
    *,
    damping: ThreeBodyDamping = ZeroThreeBodyD3(),
    rvdw_table: Tensor | TableFunction | None = None,
    r4r2_table: Tensor | TableFunction | None = None,
    cutoff: float = defaults.D3_DISP3_CUTOFF,
    width: float = defaults.D3_DISP3_WIDTH,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term of a molecule, over all triples of
    atoms. This takes ``O(nat**3)`` memory, whatever the cutoff.

    Parameters
    ----------
    structure : Structure
        The system, a molecule: atomic numbers and Cartesian coordinates,
        single or batched (see :func:`tad_dftd3.disp.dftd3`).
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    param : DampingParam
        DFT-D3 damping parameters, read by `damping`.
    damping : ThreeBodyDamping, optional
        The damping. Defaults to zero damping.
    rvdw_table : Tensor | TableFunction | None, optional
        Van der Waals radii per element pair, indexed by atomic numbers, of
        shape ``(104, 104)``, or ``None`` for
        :func:`tad_mctc.data.radii.VDW_PAIRWISE`.
    r4r2_table : Tensor | TableFunction | None, optional
        r⁴ over r² expectation values per element, of shape ``(119,)``, or
        ``None`` for :func:`tad_dftd3.data.R4R2`. Both tables give the radii
        of the pairs, see :func:`pair_radii`.
    cutoff : float, optional
        Real-space cutoff, in Bohr. A triple is dropped as a whole if one of
        its three distances is beyond it. Defaults to
        :data:`tad_dftd3.defaults.D3_DISP3_CUTOFF`.
    width : float, optional
        Width of the smooth cutoff, in Bohr: a triple is scaled by the switch
        of each of its three distances, see :mod:`tad_dftd3.cutoff`. Defaults
        to zero, a hard cutoff.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.

    Raises
    ------
    ValueError
        If `structure` is a periodic cell, see :func:`dispersion_atm_periodic`.
    """
    if structure.lattice is not None:
        raise ValueError(
            "'structure' is a periodic cell; use 'dispersion_atm_periodic'."
        )

    numbers, positions = structure.numbers, structure.positions
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    cutoff2 = cutoff * cutoff
    radii = pair_radii(structure, rvdw_table, r4r2_table)
    rdamp, rvdw = radii[..., 0], radii[..., 1]

    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_self=True)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    one = torch.tensor(1.0, **dd)

    # C9_ABC = sqrt(|C6_AB * C6_AC * C6_BC|), scaled by `s9` in the damping
    c9 = storch.safe_sqrt(
        torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3))
    )

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
    # upon exponentiation in the damping, so the triples it sees are finite
    # and positive everywhere (ones where masked), see `TripleData`
    r2pairs = torch.where(mask_pairs, distances, one)
    triples = TripleData(
        r=torch.where(mask_triples, r1, one),
        r2ij=r2pairs.unsqueeze(-1),
        r2ik=r2pairs.unsqueeze(-2),
        r2jk=r2pairs.unsqueeze(-3),
        rdampij=rdamp.unsqueeze(-1),
        rdampik=rdamp.unsqueeze(-2),
        rdampjk=rdamp.unsqueeze(-3),
        rvdwij=rvdw.unsqueeze(-1),
        rvdwik=rvdw.unsqueeze(-2),
        rvdwjk=rvdw.unsqueeze(-3),
    )

    # to fix the previous mask, we mask again (not strictly necessary because
    # `ang` is also masked and we later multiply with `ang`)
    fdamp = torch.where(mask_triples, damping(triples, param), zero)

    s = torch.where(
        mask_triples,
        (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik),
        zero,
    )

    ang = torch.where(
        mask_triples
        * (r2ij <= cutoff2)
        * (r2ik <= cutoff2)
        * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        zero,
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


def pair_radii(
    structure: Structure,
    rvdw_table: Tensor | TableFunction | None,
    r4r2_table: Tensor | TableFunction | None,
) -> Tensor:
    """
    The two radii D3 supplies for every pair of atoms, stacked on a last axis
    of length two, ``(..., nat, nat, 2)``: the damping radius
    ``sqrt(3 r4r2_i r4r2_j)`` (as D4, read by the dampings ported from dftd)
    and the van-der-Waals radius (read by the zero damping of s-dftd3). Both
    are one for padding atoms, see :class:`~tad_dftd3.damping.TripleData`.
    """
    numbers, positions = structure.numbers, structure.positions

    r4r2 = element_table(r4r2_table, "r4r2_table", positions)[numbers]
    r4r2 = _nonzero(r4r2)
    rdamp = torch.sqrt(3.0 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2))

    table = element_table(rvdw_table, "rvdw_table", positions)
    rvdw = _nonzero(table[numbers.unsqueeze(-1), numbers.unsqueeze(-2)])

    return torch.stack((rdamp, rvdw), dim=-1)


def atm_term(
    r2ij: Tensor,
    r2ik: Tensor,
    r2jk: Tensor,
    radii: tuple[Tensor, Tensor, Tensor],
    damping: ThreeBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
    valid: Tensor | None = None,
) -> Tensor:
    """
    Angular part, damping (with `s9`) and switch of triples, without the
    ``sqrt(|c6 c6 c6|)``, from their squared side lengths and the `radii` of
    their pairs ij, ik and jk (each stacked as by :func:`pair_radii`), which
    must be positive. Zero for a triple that is not `valid` or has a side
    beyond `cutoff`: a triple is kept or dropped as a whole, like s-dftd3.

    The sides of a dropped triple are replaced before any square root, so
    the derivatives stay finite whatever the caller's enumeration holds.
    """
    cutoff2 = cutoff * cutoff
    mask = (r2ij <= cutoff2) & (r2ik <= cutoff2) & (r2jk <= cutoff2)
    if valid is not None:
        mask = mask & valid

    one = torch.ones_like(r2ij)
    r2ij = torch.where(mask, r2ij, one)
    r2ik = torch.where(mask, r2ik, one)
    r2jk = torch.where(mask, r2jk, one)

    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    rij, rik, rjk = radii
    triples = TripleData(
        r1,
        r2ij,
        r2ik,
        r2jk,
        rij[..., 0],
        rik[..., 0],
        rjk[..., 0],
        rij[..., 1],
        rik[..., 1],
        rjk[..., 1],
    )
    fdamp = damping(triples, param)

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

    return ang * fdamp


def dispersion_atm_periodic(
    structure: Structure,
    c6: Tensor,
    param: DampingParam,
    shifts: PeriodicShifts | None = None,
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
    Axilrod-Teller-Muto dispersion term of a periodic cell: every atom ``i``
    of the cell with every pair of atoms or images of atoms within `cutoff`
    of it (also images of ``i`` itself), as s-dftd3.

    The energy of an atom is the sum over its triples divided by six: every
    triple of the crystal is counted from each of its three atoms, with its
    other two in both orders, which is its equal share of the three.

    The legs from ``i`` are the images within the cutoff, picked from the
    grid of atoms and shifts, and the pairs of them are formed in blocks of
    at most `max_triples`. So memory follows the number of images within the
    cutoff of an atom (``O(n_image**2)`` triples per atom), not the shift
    table. The pick is data dependent, so this does not survive ``vmap`` or
    ``torch.compile(fullgraph=True)``; for those, see
    :func:`tad_dftd3.sparse.dispersion_atm_sparse`.

    Parameters
    ----------
    structure : Structure
        The system, a periodic cell, see :func:`tad_dftd3.disp.dftd3`.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    param : DampingParam
        DFT-D3 damping parameters, read by `damping`.
    shifts : PeriodicShifts | None, optional
        Periodic image shifts, built at least at `cutoff`, see
        :func:`tad_dftd3.disp.dftd3`. Built here if missing.
    damping, rvdw_table, r4r2_table, cutoff, width
        See :func:`dispersion_atm`.
    max_triples : int, optional
        Upper bound on the triples held in memory at once in the forward
        pass. Defaults to ``2_000_000``.
    checkpoint : bool, optional
        Recompute each block in the backward pass instead of keeping its
        intermediates. Any derivative order, but not ``vmap``. Defaults to
        ``False``.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.

    Raises
    ------
    ValueError
        If `structure` has no lattice, or `shifts` does not cover it at
        `cutoff`.
    """
    numbers, positions = structure.numbers, structure.positions
    lattice, periodic = structure.lattice, structure.periodic
    if lattice is None or periodic is None:
        raise ValueError("'structure' has no lattice; use 'dispersion_atm'.")

    if shifts is None:
        shifts = build_periodic_shifts(lattice, periodic, cutoff)
    else:
        shifts.check_compatible(structure, cutoff)

    radii = pair_radii(structure, rvdw_table, r4r2_table)

    # as for the pair terms: the atoms folded into the central cell, which
    # the shifts are built for, the translation of each image and which
    # `(i, j, image)` are real pairs
    images = _periodic_images(
        numbers, positions, lattice, shifts.shifts, periodic
    )
    assert images.translations is not None

    nat, n_shift = positions.shape[-2], shifts.shifts.shape[0]
    folded = images.positions.reshape(-1, nat, 3)
    n_system = folded.shape[0]
    translations = images.translations.reshape(-1, n_shift, 3)
    translations = translations.expand(n_system, n_shift, 3)
    valid = images.valid.reshape(n_system, nat, nat, n_shift)
    valid = valid.expand(n_system, nat, nat, n_shift)

    c6_systems = c6.reshape(n_system, nat, nat)
    radii_systems = radii.reshape(n_system, nat, nat, 2)

    cutoff2 = cutoff * cutoff
    eps = torch.finfo(positions.dtype).eps
    zero = torch.zeros((), device=positions.device, dtype=positions.dtype)

    energies = []
    for system in range(n_system):
        c6_sys, radii_sys = c6_systems[system], radii_systems[system]

        for i in range(nat):
            # (nat, n_shift, 3): from atom `i` to every image of every atom
            vectors = (
                folded[system].unsqueeze(1)
                - folded[system, i]
                + translations[system].unsqueeze(0)
            )
            r2 = vectors.pow(2).sum(-1)

            with torch.no_grad():
                keep = valid[system, i] & (r2 <= cutoff2) & (r2 >= eps)
                atoms, shift_idx = torch.nonzero(keep, as_tuple=True)

            legs = vectors.reshape(-1, 3).index_select(
                0, atoms * n_shift + shift_idx
            )
            c6_i = c6_sys[i].index_select(0, atoms)
            radii_i = radii_sys[i].index_select(0, atoms)

            # At least one block, empty for an atom without legs, so that the
            # energy is still part of the graph of the positions (with a zero
            # gradient), also if no atom has any.
            n_leg = atoms.shape[0]
            energy = zero
            block = max(1, max_triples // max(n_leg, 1))
            for start in range(0, max(n_leg, 1), block):
                rows = slice(start, start + block)
                args = (
                    legs[rows],
                    legs,
                    atoms[rows],
                    atoms,
                    c6_i[rows],
                    c6_i,
                    radii_i[rows],
                    radii_i,
                    c6_sys,
                    radii_sys,
                    damping,
                    param,
                    cutoff,
                    width,
                )
                if checkpoint:
                    energy = energy + _torch_checkpoint(
                        _periodic_block_energy, *args, use_reentrant=False
                    )
                else:
                    energy = energy + _periodic_block_energy(*args)

            energies.append(energy)

    return torch.stack(energies).reshape(numbers.shape)


def _periodic_block_energy(
    vec_a: Tensor,
    vec_b: Tensor,
    atom_a: Tensor,
    atom_b: Tensor,
    c6_a: Tensor,
    c6_b: Tensor,
    radii_a: Tensor,
    radii_b: Tensor,
    c6: Tensor,
    radii: Tensor,
    damping: ThreeBodyDamping,
    param: DampingParam,
    cutoff: float,
    width: float,
) -> Tensor:
    """
    Sum of the ATM energy of the triples of one atom ``i`` of the cell with
    each of the ``A`` images in ``a`` and each of the ``B`` in ``b``, over
    the ``A * B`` combinations, divided by six (the dense sum runs over the
    six orderings of a triple).

    `vec_a` and `vec_b` are the vectors from ``i`` to its images, `atom_*`
    the atoms they are images of, and `c6_*` and `radii_*` the C6 and the
    radii (see :func:`pair_radii`) of ``i`` with them. `c6` and `radii` are
    the tables of the system.
    """
    vec_jk = vec_b.unsqueeze(0) - vec_a.unsqueeze(1)

    r2ij = vec_a.pow(2).sum(-1).unsqueeze(1)
    r2ik = vec_b.pow(2).sum(-1).unsqueeze(0)
    r2jk = vec_jk.pow(2).sum(-1)

    c6_jk = c6[atom_a.unsqueeze(1), atom_b.unsqueeze(0)]
    radii_jk = radii[atom_a.unsqueeze(1), atom_b.unsqueeze(0)]

    c9 = storch.safe_sqrt(
        torch.abs(c6_a.unsqueeze(1) * c6_b.unsqueeze(0) * c6_jk)
    )

    # An image of an atom with itself at the zero shift is excluded by the
    # caller for the legs from `i`, but two images may still coincide.
    eps = torch.finfo(vec_a.dtype).eps
    term = atm_term(
        r2ij,
        r2ik,
        r2jk,
        (radii_a.unsqueeze(1), radii_b.unsqueeze(0), radii_jk),
        damping,
        param,
        cutoff,
        width,
        r2jk >= eps,
    )

    return torch.sum(term * c9) / 6.0


def _nonzero(x: Tensor) -> Tensor:
    """`x`, with ones in place of zeros (the radii of padding atoms)."""
    return torch.where(x != 0, x, torch.ones_like(x))
