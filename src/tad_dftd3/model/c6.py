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
  output with a single GEMM (see ``_atomic_c6_fast``). Measured against the
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

* **"safe"** -- the same bilinear form as "fast" (see ``_atomic_c6_fast``),
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
tracing (checked via `_is_batched_anywhere`/`is_compiling`, in that order --
`is_compiling` short-circuits before the `vmap`-only check ever runs, which
matters because that check is itself not traceable by `torch.compile`).
`_is_batched_anywhere` inspects `numbers` itself rather than any global
transform state, which matters in two ways: it stays on "fast" when a
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
from tad_mctc._version import __tversion__
from tad_mctc.autograd import is_batched
from tad_mctc.math import einsum
from tad_mctc.typing import Callable, Tensor

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
    if is_compiling() or _is_batched_anywhere(numbers):
        return _atomic_c6_safe(numbers, weights, reference)

    return _atomic_c6_fast(numbers, weights, reference)


# "fast" path -- groups atoms by the elements actually present. Not
# vmap/compile-safe: `torch.unique` has data-dependent output shape. Only
# ever called when neither `is_compiling()` nor `_is_batched_anywhere` is
# true, so a shared, global `numbers.unique()` across any leading batch
# dimensions (e.g. a `tad_mctc.batch.pack`-ed multi-molecule batch) is both
# safe to call and correct: atoms from a molecule that happens to lack one
# of the globally-present elements simply get a zero contribution there.


def _atomic_c6_fast(
    numbers: Tensor, weights: Tensor, reference: Reference
) -> Tensor:
    """
    Bilinear form factored over the elements actually present in `numbers`
    (see module docstring for the derivation and measured speedup).
    """
    unique_numbers = torch.unique(numbers)

    # Gather reference C6 blocks against only the unique elements present:
    # (..., nelements, nelements, 7, 7) -> (..., nat, nunique, 7, 7)
    rc6 = reference.c6[numbers.unsqueeze(-1), unique_numbers]

    # Contract the reference block with the weights of the first atom.
    # g[..., i, u, b] = sum_a w[..., i, a] * rc6[..., i, u, a, b]
    # (..., nat, nunique, 7, 7) * (..., nat, 7) -> (..., nat, nunique, 7)
    g = einsum("...iuab,...ia->...iub", rc6, weights)

    # Weights of the second atom, scattered onto the unique-element axis:
    # only the entry matching the atom's own element is non-zero.
    # (..., nat, 7) * (..., nat, nunique) -> (..., nat, nunique, 7)
    onehot = (numbers.unsqueeze(-1) == unique_numbers).type(weights.dtype)
    wu = weights.unsqueeze(-2) * onehot.unsqueeze(-1)

    # Final contraction is a single (batched) matrix multiplication over the
    # flattened (nunique * 7) axis, directly producing the dense (nat, nat)
    # output -- no (..., nat, nat, ...) intermediate is ever built.
    # (..., nat, nunique, 7) * (..., nat, nunique, 7) -> (..., nat, nat)
    return einsum("...iub,...jub->...ij", g, wu)


# "safe" path -- same bilinear form as "fast" (see module docstring), but
# grouping by the fixed, data-independent set of all possible elements
# instead of `numbers.unique()`. vmap- and compile-safe: no `unique`/
# `nonzero` anywhere, and `reference.c6.shape[0]` (nelements) is a plain
# Python int, not traced. Supports an arbitrary leading "..." batch
# dimension directly via einsum, same as "fast" -- no per-geometry loop
# needed once grouping no longer depends on which elements are present.


def _atomic_c6_safe(
    numbers: Tensor, weights: Tensor, reference: Reference
) -> Tensor:
    """
    Bilinear form factored over the fixed set of all possible elements (see
    module docstring for the derivation and the memory/FLOPs trade-off).
    """
    all_elements = torch.arange(reference.c6.shape[0], device=numbers.device)

    # Gather reference C6 blocks against every possible element:
    # (..., nelements, nelements, 7, 7) -> (..., nat, nelements, 7, 7)
    rc6 = reference.c6[numbers.unsqueeze(-1), all_elements]

    # Contract the reference block with the weights of the first atom.
    # g[..., i, u, b] = sum_a w[..., i, a] * rc6[..., i, u, a, b]
    # (..., nat, nelements, 7, 7) * (..., nat, 7) -> (..., nat, nelements, 7)
    g = _einsum_compile_safe("...iuab,...ia->...iub", rc6, weights)

    # Weights of the second atom, scattered onto the element axis: only the
    # entry matching the atom's own element is non-zero.
    # (..., nat, 7) * (..., nat, nelements) -> (..., nat, nelements, 7)
    onehot = (numbers.unsqueeze(-1) == all_elements).type(weights.dtype)
    wu = weights.unsqueeze(-2) * onehot.unsqueeze(-1)

    # Final contraction is a single (batched) matrix multiplication over the
    # flattened (nelements * 7) axis, directly producing the dense (nat, nat)
    # output -- no (..., nat, nat, ...) intermediate is ever built.
    # (..., nat, nelements, 7) * (..., nat, nelements, 7) -> (..., nat, nat)
    return _einsum_compile_safe("...iub,...jub->...ij", g, wu)


def _resolve_is_batched_anywhere() -> Callable[[Tensor], bool]:
    """
    Resolve, once, how to walk the functorch wrapper stack on this PyTorch.

    `is_batched` (from `tad_mctc`, pinned at 0.8.0) only inspects the
    *outermost* wrapper layer: a tensor that is genuinely `vmap`-batched,
    but additionally wrapped by an *outer* `torch.func.jacrev`/`grad`
    transform (e.g. under ``vmap(jacrev(jacrev(f)))``, as used by
    `tad_mctc.autograd.hess_fn_rev` for Hessians), reports as *not* batched
    at that outer layer even though a `vmap` level exists further down the
    wrapper stack. This walks that stack (unwrapping through grad-tracking
    layers) to answer "is this tensor batched anywhere" -- which is what
    actually determines whether `numbers.unique()` is safe to call.

    ``is_functorch_wrapped_tensor``/``get_unwrapped`` are less central
    functorch APIs than ``is_batchedtensor``, and this project's own CI
    matrix exercises PyTorch versions old enough (down to 2.0.1) that their
    presence should not be assumed. If either is missing, fall back to
    `is_batched` (single layer): still correct whenever no grad-tracking
    layer obscures the batched one, and strictly better than raising
    `AttributeError`.
    """
    if __tversion__ < (2, 0, 0):
        return lambda x: False

    ft = torch._C._functorch  # pyright: ignore[reportAttributeAccessIssue]
    is_wrapped = getattr(ft, "is_functorch_wrapped_tensor", None)
    get_unwrapped = getattr(ft, "get_unwrapped", None)

    if is_wrapped is None or get_unwrapped is None:  # pragma: no cover
        return is_batched

    def _walk(x: Tensor) -> bool:
        while is_wrapped(x):
            if ft.is_batchedtensor(x):
                return True
            x = get_unwrapped(x)
        return False

    return _walk


_is_batched_anywhere = _resolve_is_batched_anywhere()


# `is_compiling`, ported from `tad_mctc.tools.compile` -- not yet in a
# released tad-mctc (tad-dftd3 pins `tad-mctc==0.8.0`, which predates it).
# Remove this copy and import from `tad_mctc.tools` once a tad-mctc release
# with `is_compiling` is pinned.


def _always_false() -> bool:  # pragma: no cover
    """``torch.compile`` does not exist, or exposes no way to ask."""
    return False


def _resolve_is_compiling() -> Callable[[], bool]:
    """
    Resolve, once, which underlying "is compiling" query this PyTorch
    offers.

    This capability probe -- ``getattr``/``hasattr`` chains, an explicit
    ``import torch._dynamo`` -- has to run here, outside of
    :func:`is_compiling`'s own body: ``is_compiling()`` is called from
    `atomic_c6`, which may itself get traced under
    ``torch.compile(fullgraph=True)``, and Dynamo cannot trace
    ``hasattr``/``getattr`` introspection on a module object
    (``Unsupported: hasattr: PythonModuleVariable()``); it can trace a
    plain call to a resolved function just fine.

    Neither the public ``torch.compiler.is_compiling`` nor its older,
    private predecessor ``torch._dynamo.is_compiling`` reliably exists (or
    resolves without raising) purely as a function of ``__tversion__``:
    ``torch.compiler`` can exist without yet having ``is_compiling`` on it
    (added later than the module itself), and ``torch._dynamo`` -- even on
    a version that ships it -- is only exposed as a ``torch`` attribute
    once something has imported it, which nothing upstream of this call is
    guaranteed to have done.
    """
    try:
        import torch._dynamo as _torch_dynamo  # noqa: F401  # pylint: disable=unused-import, protected-access
    except ImportError:  # pragma: no cover
        # Only unavailable on PyTorch < 2.0, not exercised by any single
        # CI job's torch version; the other probes below fall through.
        pass

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return compiler.is_compiling

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return dynamo.is_compiling

    return _always_false  # pragma: no cover


_is_compiling_impl = _resolve_is_compiling()


def is_compiling() -> bool:
    """Whether we are currently being traced by ``torch.compile``."""
    return bool(_is_compiling_impl())


def _einsum_compile_safe(equation: str, *operands: Tensor) -> Tensor:
    """
    ``einsum``, bypassing ``tad_mctc.math.einsum``'s ``opt_einsum`` backend
    while tracing.

    ``tad_mctc.math.einsum`` already means to skip ``opt_einsum.contract``
    under ``torch.compile`` -- Dynamo cannot trace its Python-side
    contraction-path bookkeeping -- gated on its own, internal
    ``tad_mctc.tools.is_compiling`` check. The tad-mctc release currently
    pinned by tad-dftd3 (``tad-mctc==0.8.0``) predates a fix to that check,
    so on newer PyTorch it can under-report, silently defeating the guard it
    exists to provide (``opt_einsum``'s ``threading.get_ident()`` call is
    unsupported by Dynamo and raises). `_atomic_c6_safe` is called
    specifically because it must survive `torch.compile`, so it routes
    around `tad_mctc.math.einsum`'s gate entirely and uses this module's own
    `is_compiling` (verified correct against the exact PyTorch version this
    was hit on) instead: plain `torch.einsum` while tracing --
    `opt_einsum`'s contraction-order optimisation buys nothing once Inductor
    does its own fusion anyway -- and `tad_mctc.math.einsum` otherwise, to
    keep the optimisation eager execution still benefits from.
    """
    if is_compiling():
        return torch.einsum(equation, *operands)
    return einsum(equation, *operands)
