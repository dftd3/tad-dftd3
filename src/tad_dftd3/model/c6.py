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
  output with a single GEMM (see ``_bilinear``). Measured against the
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

* **"safe"** -- the same bilinear form as "fast" (see ``_bilinear``),
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

`atomic_c6` picks "fast" unless `numbers` is `vmap`-batched or `compile` is
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
"""

from __future__ import annotations

import torch
from tad_mctc.autograd import is_vmapped
from tad_mctc.typing import Tensor

from ..reference import Reference

__all__ = ["atomic_c6"]


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
    if is_vmapped(numbers):
        return _atomic_c6_safe(numbers, weights, reference)

    return _atomic_c6_fast(numbers, weights, reference)


def _atomic_c6_fast(
    numbers: Tensor, weights: Tensor, reference: Reference
) -> Tensor:
    """
    Bilinear form factored over the elements actually present in `numbers`
    (see module docstring for the derivation and measured speedup).

    Not vmap/compile-safe: `torch.unique` has data-dependent output shape.
    Only ever called when `is_vmapped` (also true while compiling) is false,
    so a shared, global `numbers.unique()` across any leading batch
    dimensions (e.g. a `tad_mctc.batch.pack`-ed multi-molecule batch) is
    correct: atoms from a molecule that lacks one of the globally-present
    elements simply get a zero contribution there.
    """
    return _bilinear(numbers, weights, reference, torch.unique(numbers))


def _atomic_c6_safe(
    numbers: Tensor, weights: Tensor, reference: Reference
) -> Tensor:
    """
    Bilinear form factored over the fixed set of all possible elements (see
    module docstring for the derivation and the memory/FLOPs trade-off).

    vmap- and compile-safe: no `unique`/`nonzero`, and `reference.c6.shape[0]`
    is a plain Python int, not traced.
    """
    elements = torch.arange(reference.c6.shape[0], device=numbers.device)
    return _bilinear(numbers, weights, reference, elements)


def _bilinear(
    numbers: Tensor, weights: Tensor, reference: Reference, elements: Tensor
) -> Tensor:
    """Bilinear form of the reference C6 over the given element axis."""
    # Gather reference C6 blocks against only the given elements:
    # (..., nelements, nelements, 7, 7) -> (..., nat, nel, 7, 7)
    rc6 = reference.c6[numbers.unsqueeze(-1), elements]

    # Contract the reference block with the weights of the first atom.
    # g[..., i, u, b] = sum_a w[..., i, a] * rc6[..., i, u, a, b]
    # (..., nat, 7) * (..., nat, nel, 7, 7) -> (..., nat, nel, 7)
    #
    # The operand order is deliberate: `torch.einsum("...iuab,...ia->...iub",
    # rc6, weights)` is up to ~3x slower for many elements, because
    # `torch.einsum` lays out its `bmm` according to the operand order.
    # `tad_mctc.math.einsum` (`opt_einsum`) picks the fast order itself, but
    # adds overhead (up to ~4x slower for few atoms) and cannot be traced by
    # `torch.compile`.
    g = torch.einsum("...ia,...iuab->...iub", weights, rc6)

    # Weights of the second atom, scattered onto the element axis: only the
    # entry matching the atom's own element is non-zero.
    # (..., nat, 7) * (..., nat, nel) -> (..., nat, nel, 7)
    onehot = (numbers.unsqueeze(-1) == elements).type(weights.dtype)
    wu = weights.unsqueeze(-2) * onehot.unsqueeze(-1)

    # Final contraction is a single (batched) matrix multiplication over the
    # flattened (nel * 7) axis, directly producing the dense (nat, nat)
    # output -- no (..., nat, nat, ...) intermediate is ever built. An
    # explicit `matmul` is up to ~2x faster than `einsum("...iub,...jub->...ij")`.
    return g.flatten(-2) @ wu.flatten(-2).transpose(-1, -2)
