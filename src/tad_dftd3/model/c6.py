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
Model: Atomic C6
================

Computation of atomic C6 dispersion coefficients.

A naive evaluation gathers ``reference.c6[Z_i, Z_j]`` for every atom *pair*,
materialising a dense ``(..., nat, nat, 7, 7)`` tensor -- 1.5 GB at 2000
atoms. ``C6ref`` only depends on the *element* pair, not the atom pair, so
grouping atoms by element before contracting against the table avoids ever
materialising that tensor. There are two ways to do that grouping, and this
module picks between them at call time:

* **"fast"** -- group by the elements actually present, via
  ``numbers.unique()``. The reference block is gathered only against those
  ``nunique`` elements (``(..., nat, nunique, 7, 7)``, linear in ``nat``,
  ``nunique`` typically a handful) and reduced to the ``(..., nat, nat)``
  output with a single GEMM (see ``AtomicC6.dense``). Measured against the
  naive dense evaluation, on CPU (float64, single-threaded):

  ==== =======  ===================  ==========================
   nat nunique  intermediate memory  wall time (mean of 5 runs)
  ==== =======  ===================  ==========================
   500       4      98.0 -> 0.8 MB   125x smaller,  167x faster
   500      60      98.0 -> 11.8 MB    8x smaller,   20x faster
  2000       4    1568.0 -> 3.1 MB   500x smaller,  256x faster
  2000      60    1568.0 -> 47.0 MB   33x smaller,   22x faster
  ==== =======  ===================  ==========================

  ``numbers.unique()``'s output shape is data-dependent, which is illegal
  under `torch.func.vmap` (it cannot batch an op whose output shape could
  differ per batch element) and forces a graph break under
  `torch.compile`. This path is therefore only used when neither applies.

* **"safe"** -- the same bilinear form as "fast" (see ``_factors``),
  but grouping by the fixed, data-*independent* set of all ~104 possible
  elements instead of ``numbers.unique()``. No ``unique``/``nonzero``
  anywhere, so it survives both `vmap` and `compile`. Its intermediates are
  ``(..., nat, nelements, 7)`` -- linear in ``nat``, like "fast" -- rather
  than the ``(..., nat, nat, 7)`` a naive gather-then-reduce would need; only
  the final ``(..., nat, nat)`` output itself is quadratic, which is
  unavoidable for a dense result. The trade for never materialising that
  ``(..., nat, nat, 7)`` intermediate is FLOPs: the closing contraction is a
  single GEMM over the full ``nelements`` axis (multiplying through the
  zeros a one-hot mask leaves in most of it) rather than a 7-wide
  elementwise reduction over a pre-gathered, pair-sized tensor -- roughly
  ``nelements`` times more arithmetic, worthwhile because GEMM throughput is
  cheap relative to the memory this avoids.

`factored_c6` (and with it `atomic_c6`) picks "fast" unless `numbers` is `vmap`-batched or `compile` is
tracing, both answered by `tad_mctc.autograd.is_vmapped`, which is always
true while `compile` traces. It inspects `numbers` itself rather than any
global transform state, which matters in two ways: it stays on "fast" when a
`vmap` is active but `numbers` is not the batched argument (e.g. `vmap`
over positions only), and -- since neither path here uses a custom
`torch.autograd.Function` with `ctx.save_for_backward` -- it never has to
account for a save/restore round trip silently dropping a `BatchedTensor`
wrapper while the transform is still active; plain composite ops keep the
same tensor object (and its wrapper) throughout, so the ordinary forward
call is the only place this ever needs checking.

`factored_c6` stops before the closing GEMM: its :class:`AtomicC6` holds the
per-atom factors, linear in ``nat``, and gives the C6 of single pairs, which
is what the sum over a neighbour list needs (see :mod:`tad_dftd3.sparse`).
"""

from __future__ import annotations

import torch
from tad_mctc.autograd import is_functorch_tensor, is_vmapped
from tad_mctc.tree import Node, child
from tad_mctc.typing import Tensor

from ..reference import Reference

__all__ = ["AtomicC6", "atomic_c6", "factored_c6"]


class AtomicC6(Node):
    """
    The C6 coefficients of all pairs of atoms, factored per atom: the
    reference C6 contracted with the weights of the first atom, against
    each element, and the weights of the second atom, which pick their
    element's entry.

    ``dense()`` is the ``(..., nat, nat)`` matrix of :func:`atomic_c6`, and
    ``pair(idx_i, idx_j)`` the coefficients of given pairs only, which is
    all a neighbour list needs: nothing of size ``nat**2`` is ever built,
    neither in the forward nor in the backward pass (where a lookup in a
    dense matrix creates a gradient of its full size, per chunk of pairs).

    Built by :func:`factored_c6`.

    Parameters
    ----------
    partial : Tensor
        ``sum_a w[i, a] * c6ref[Z_i, u, a, b]`` of every atom ``i`` with
        every element ``u`` of the element axis, ``(..., nat, nel, nref)``.
    weights : Tensor
        Weights of the reference systems of every atom, ``(..., nat, nref)``.
        ``nref`` is at most 7, and fewer where the elements present have
        fewer references (see :func:`factored_c6`).
    element : Tensor
        Index of the element of every atom on the element axis of
        `partial`, ``(..., nat)``.
    """

    partial: Tensor = child()
    weights: Tensor = child()
    element: Tensor = child()

    def dense(self) -> Tensor:
        """
        The C6 coefficients of all pairs of atoms, ``(..., nat, nat)``.

        Returns
        -------
        Tensor
            The C6 matrix, as :func:`atomic_c6`.
        """
        # Weights of the second atom, scattered onto the element axis: only
        # the entry matching the atom's own element is non-zero.
        # (..., nat, nref) * (..., nat, nel) -> (..., nat, nel, nref)
        elements = torch.arange(
            self.partial.shape[-2], device=self.element.device
        )
        onehot = self.element.unsqueeze(-1) == elements
        wu = (
            self.weights.unsqueeze(-2)
            * onehot.type(self.weights.dtype)[..., None]
        )

        # Final contraction is a single (batched) matrix multiplication over
        # the flattened (nel * nref) axis, directly producing the dense
        # (nat, nat) output -- no (..., nat, nat, ...) intermediate is ever
        # built. An explicit `matmul` is up to ~2x faster than
        # `einsum("...iub,...jub->...ij")`.
        return self.partial.flatten(-2) @ wu.flatten(-2).transpose(-1, -2)

    def pair(self, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """
        The C6 coefficients of the pairs ``(idx_i, idx_j)``, with the atoms of
        a batch numbered as one flat system, atom ``i`` of system ``b`` being
        ``b * nat + i`` (as in a :class:`~tad_mctc.neighbor.list.NeighborList`).

        Parameters
        ----------
        idx_i, idx_j : Tensor
            Flat indices of the two atoms of each pair, ``(n,)``.

        Returns
        -------
        Tensor
            C6 coefficients of the pairs, ``(n,)``.
        """
        nel, nref = self.partial.shape[-2:]

        # `index_select` on flat views: the gradient of a lookup is the size
        # of the table it reads, here linear in the number of atoms.
        partial = self.partial.reshape(-1, nref)
        element_j = self.element.reshape(-1).index_select(0, idx_j)
        rows = partial.index_select(0, idx_i * nel + element_j)

        weights_j = self.weights.reshape(-1, nref).index_select(0, idx_j)
        return torch.sum(rows * weights_j, dim=-1)


def factored_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
) -> AtomicC6:
    """
    Atomic dispersion coefficients, factored per atom, see :class:`AtomicC6`.

    Outside ``vmap`` and ``torch.compile``, the factors are formed over the
    elements present only and over the references these have, up to the
    largest number of them (5 for H, C, N and O, out of 7): only the first
    references of an element are defined, and the C6 of the others is zero.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system of shape `(..., nat)`.
    weights : Tensor
        Weights of all reference systems of shape `(..., nat, 7)`.
    reference : Reference
        Reference systems for D3 model.

    Returns
    -------
    AtomicC6
        The factored C6 coefficients of all pairs of atoms.
    """
    if is_vmapped(numbers):
        # "safe": all elements, the atomic number is the index
        elements = torch.arange(reference.c6.shape[0], device=numbers.device)
        return _factors(numbers, weights, reference, elements, numbers)

    # "fast": the elements present, sorted by `unique`. Not vmap/compile-safe
    # (data-dependent shape), so only when neither applies, and then shared
    # over any leading batch dimensions: an atom of a molecule that lacks one
    # of the elements simply gets a zero contribution there.
    elements = torch.unique(numbers)
    element = torch.searchsorted(elements, numbers.contiguous())

    # Each pair of atoms is a sum over the references of both, but only the
    # first few references of an element are defined (H 2, C 5, O 3, ...)
    # and the C6 of the others is zero: cut where that holds for every
    # element present. Exact, and it shrinks every per-pair lookup of the
    # factors, and its gradient, by as much.
    width = _reference_width(reference, elements)
    return _factors(
        numbers, weights[..., :width], reference, elements, element, width
    )


def atomic_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
) -> Tensor:
    """
    Calculate atomic dispersion coefficients.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system of shape `(..., nat)`.
    weights : Tensor
        Weights of all reference systems of shape `(..., nat, 7)`.
    reference : Reference
        Reference systems for D3 model. Contains the reference C6 coefficients
        of shape `(..., nelements, nelements, 7, 7)`.

    Returns
    -------
    Tensor
        Atomic dispersion coefficients of shape `(..., nat, nat)`.
    """
    return factored_c6(numbers, weights, reference).dense()


def _reference_width(reference: Reference, elements: Tensor) -> int:
    """
    The number of leading references that hold a non-zero C6 for any pair of
    `elements`; beyond it, the reference C6 of all of them is zero. Reads the
    values of the table, so the full width is kept while the table itself is
    being transformed (e.g. differentiated with ``torch.func``).
    """
    nref = reference.c6.shape[-1]
    if is_functorch_tensor(reference.c6):
        return nref

    block = reference.c6[elements.unsqueeze(-1), elements]
    used = (block != 0).flatten(0, 1).any(0)
    used = used.any(0) | used.any(1)
    if not bool(used.any()):
        return 1
    return int(used.nonzero().max()) + 1


def _factors(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
    elements: Tensor,
    element: Tensor,
    width: int | None = None,
) -> AtomicC6:
    """
    The factors of the bilinear form of the reference C6 over the given
    element axis, `element` being the index of each atom's element on it,
    and over the first `width` references (all if ``None``), which `weights`
    already has.
    """
    # Gather reference C6 blocks against only the given elements:
    # (..., nelements, nelements, 7, 7) -> (..., nat, nel, width, width)
    table = reference.c6
    if width is not None:
        table = table[..., :width, :width]
    rc6 = table[numbers.unsqueeze(-1), elements]

    # Contract the reference block with the weights of the first atom.
    # g[..., i, u, b] = sum_a w[..., i, a] * rc6[..., i, u, a, b]
    # (..., nat, nref) * (..., nat, nel, nref, nref) -> (..., nat, nel, nref)
    #
    # The operand order is deliberate: `torch.einsum("...iuab,...ia->...iub",
    # rc6, weights)` is up to ~3x slower for many elements, because
    # `torch.einsum` lays out its `bmm` according to the operand order.
    # `tad_mctc.math.einsum` (`opt_einsum`) picks the fast order itself, but
    # adds overhead (up to ~4x slower for few atoms) and cannot be traced by
    # `torch.compile`.
    g = torch.einsum("...ia,...iuab->...iub", weights, rc6)

    return AtomicC6(partial=g, weights=weights, element=element)
