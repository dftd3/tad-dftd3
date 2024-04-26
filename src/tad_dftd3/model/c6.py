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

Since this part can be the most memory-intensive, we provide a custom backward
function (i.e., analytical gradient) and options for chunking.
"""
from __future__ import annotations

import torch
from tad_mctc._version import __tversion__
from tad_mctc.math import einsum
from tad_mctc.tools import memory

from ..reference import Reference
from ..typing import Callable, Protocol, Tensor

__all__ = ["atomic_c6"]


# main entry point


def atomic_c6(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
    chunk_size: None | int = None,
) -> Tensor:
    """
    Calculate atomic dispersion coefficients.

    .. warning::

        This function is the most memory intensive part of the calculation and
        may require chunking for large systems. Without chunking, for example,
        2000 atoms (`numbers`) require the construction of a 1.5 GB tensor.

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
    _check_memory(numbers, weights, chunk_size)

    AtomicC6 = AtomicC6_V1 if __tversion__ < (2, 0, 0) else AtomicC6_V2
    res = AtomicC6.apply(numbers, weights, reference, chunk_size)
    assert res is not None
    return res


# helpers


def _check_memory(
    numbers: Tensor, weights: Tensor, chunk_size: None | int = None
) -> None:
    """
    Check memory usage for the construction of the C6 tensor.
    Throw an error or warning for potential memory issues.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    weights : Tensor
        Weights of all reference systems.
    chunk_size : None | int, optional
        Chunk size for the calculation of the C6 tensor. Defaults to `None`.

    Raises
    ------
    MemoryError
        If the estimated memory usage exceeds the total available memory.
    """
    # Required memory for the C6 tensor
    if chunk_size is None:
        size = (numbers.shape[-1], numbers.shape[-1], 7, 7)
    else:
        size = (numbers.shape[-1], chunk_size, 7, 7)
    mem = memory.memory_tensor(size, weights.dtype)

    # actual memory usage
    free, total = memory.memory_device(numbers.device)

    if mem > total:
        raise MemoryError(
            f"Estimated memory usage exceeds total available memory: {mem:.2f} "
            f"MB > {total:.2f} MB. During the construction of the C6 "
            f"dispersion coefficients, a 4D tensor of shape {size} is required "
            "for efficient tensor operations. To fit the tensor into memory, "
            "try using a chunk size or reduce the chunk size via the optional "
            "`chunk_size` argument."
        )

    if mem > free:
        # pylint: disable=import-outside-toplevel
        from warnings import warn

        warn(
            "Estimated memory usage appears to exceed the available memory: "
            f"{mem:.2f} MB > {free:.2f} MB. If the calculation fails due to "
            "memory issues, consider reducing the chunk size via the optional "
            "`chunk_size` argument.",
            ResourceWarning,
        )


def _einsum(rc6: Tensor, weights_i: Tensor, weights_j: Tensor) -> Tensor:
    """
    Perform an einsum operation for the atomic C6 coefficients.

    Parameters
    ----------
    rc6 : Tensor
        Reference C6 coefficients.
    weights_i : Tensor
        Weights of all reference systems.
    weights_j : Tensor
        Weights of all reference systems.

    Returns
    -------
    Tensor
        Atomic C6 dispersion coefficients.
    """
    # The default einsum path is fastest if the large tensors comes first.
    # (..., n1, n2, r1, r2) * (..., n1, r1) * (..., n2, r2) -> (..., n1, n2)
    return einsum(
        "...ijab,...ia,...jb->...ij",
        *(rc6, weights_i, weights_j),
        optimize=[(0, 1), (0, 1)],
    )


# full and chunked versions


def _atomic_c6_full(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
) -> Tensor:
    """
    Calculation of atomic dispersion coefficients without chunking. Might cause
    memory issues for very large systems.

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
    # NOTE: This old version creates large intermediate tensors and builds the
    # full matrix before the sum reduction, which requires a lot of memory.
    #
    # gw = w.unsqueeze(-1).unsqueeze(-3) * w.unsqueeze(-2).unsqueeze(-4)
    # c6 = torch.sum(torch.sum(torch.mul(gw, rc6), dim=-1), dim=-1)

    rc6 = reference.c6[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    return _einsum(rc6, weights, weights)


def _atomic_c6_chunked(
    numbers: Tensor,
    weights: Tensor,
    reference: Reference,
    chunk_size: int,
) -> Tensor:
    """
    Chunked version of the calculation of atomic dispersion coefficients.

    Parameters
    ----------
    numbers : Tensor
        The atomic numbers of the atoms in the system of shape `(..., nat)`.
    weights : Tensor
        Weights of all reference systems of shape `(..., nat, 7)`.
    reference : Reference
        Reference systems for D3 model. Contains the reference C6 coefficients
        of shape `(..., nelements, nelements, 7, 7)`.
    chunk_size : int
        Chunk size for the calculation of the C6 tensor.

    Returns
    -------
    Tensor
        Atomic dispersion coefficients of shape `(..., nat, nat)`.
    """

    nat = numbers.shape[-1]
    c6_output = torch.zeros(
        (*numbers.shape, nat), device=numbers.device, dtype=weights.dtype
    )

    for start in range(0, nat, chunk_size):
        end = min(start + chunk_size, nat)
        num_chunk = numbers[..., start:end]  # (..., chunk_size)

        # Chunked indexing into reference.c6: (..., chunk_size, nat, 7, 7)
        rc6_chunk = reference.c6[num_chunk.unsqueeze(-1), numbers.unsqueeze(-2)]

        # Also chunk the weights: (..., chunk_size, 7)
        weights_chunk = weights[..., start:end, :]

        # (..., n1, n2, r1, r2) * (..., n1, r1) * (..., n2, r2) -> (..., n1, n2)
        contribution = _einsum(rc6_chunk, weights_chunk, weights)

        # Add contributions to the correct slice of the output tensor
        c6_output[..., start:end, :] += contribution

    return c6_output


# custom autograd functions


class CTX(Protocol):
    save_for_backward: Callable[[Tensor, Tensor], None]
    saved_tensors: tuple[Tensor, Tensor]
    chunk_size: None | int
    reference: Reference


class AtomicC6Base(torch.autograd.Function):
    """
    Base class for the version-specific autograd function for atomic C6.
    Different PyTorch versions only require different `forward()` signatures.
    """

    @staticmethod
    def backward(ctx: CTX, grad_out: Tensor) -> tuple[None, Tensor, None, None]:
        numbers, weights = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ref = ctx.reference

        # We need the derivatives of the following expression:
        # c_ij ​= ∑a,b w_ia *× w_jb ​* c_ijab​

        ###########################
        ### Non-chunked version ###
        ###########################

        if chunk_size is None:
            # (..., nel, nel, 7, 7) -> (..., nat, nat, 7, 7)
            rc6 = ref.c6[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]

            # ∂c_ij/∂w_jb = ∑a w_ia * c_ijab
            # (..., n1, n2, r1, r2) * (..., n2, r2) -> (..., n1, n2, r2)
            g_jb = einsum("...ijab,...ia->...ijb", rc6, weights)

            # vjp: (..., n1, n2) * (..., n1, n2, r2) -> (..., n2, r2)
            _gj = einsum("...ij,...ijb->...jb", grad_out, g_jb)

            # ∂c_ij/∂w_ia = ∑b w_jb * c_ijab
            # (..., n1, n2, r1, r2) * (..., n2, r2) -> (..., n1, n2, r1)
            g_ia = einsum("...ijab,...jb->...ija", rc6, weights)

            # vjp: (..., n1, n2) * (..., n1, n2, r1) -> (..., n1, r1)
            _gi = einsum("...ij,...ija->...ia", grad_out, g_ia)

            weights_bar = _gi + _gj

            return None, weights_bar, None, None

        #######################
        ### Chunked version ###
        #######################

        nat = weights.shape[-2]
        weights_bar = torch.zeros_like(weights)

        for start in range(0, nat, chunk_size):
            end = min(start + chunk_size, nat)

            # Numbers and derivatives for this chunk
            grad_chunk = grad_out[..., start:end, :]  # (..., chunk_size, nat)
            num_chunk = numbers[..., start:end]  # (..., chunk_size)

            # Chunked indexing into reference.c6: (..., chunk_size, nat, 7, 7)
            # -> Only the "i" index is chunked!
            rc6_chunk = ref.c6[num_chunk.unsqueeze(-1), numbers.unsqueeze(-2)]

            # Also chunk the weights: (..., chunk_size, 7)
            weights_chunk = weights[..., start:end, :]

            # _gi derivative is chunked (sum over non-chunked "j" index)
            g_ia = einsum("...ijab,...jb->...ija", rc6_chunk, weights)
            _gi = einsum("...ij,...ija->...ia", grad_chunk, g_ia)

            # _gj derivative is NOT chunked (sum over chunked "i" index)
            g_jb = einsum("...ijab,...ia->...ijb", rc6_chunk, weights_chunk)
            _gj = einsum("...ij,...ijb->...jb", grad_chunk, g_jb)

            # Accumulate gradients for the current chunk
            weights_bar[..., start:end, :] += _gi
            weights_bar += _gj

        return None, weights_bar, None, None


class AtomicC6_V1(AtomicC6Base):
    """
    Custom autograd function for atomic C6 coefficients.
    This is supposed to reduce memory usage.
    """

    @staticmethod
    def forward(
        ctx: CTX,
        numbers: Tensor,
        weights: Tensor,
        reference: Reference,
        chunk_size: None | int = None,
    ) -> Tensor:
        ctx.save_for_backward(numbers, weights)
        ctx.chunk_size = chunk_size
        ctx.reference = reference

        if chunk_size is None:
            return _atomic_c6_full(numbers, weights, reference)

        return _atomic_c6_chunked(numbers, weights, reference, chunk_size)


class AtomicC6_V2(AtomicC6Base):
    """
    Custom autograd function for atomic C6 coefficients.
    This is supposed to reduce memory usage.
    """

    generate_vmap_rule = True
    # https://pytorch.org/docs/master/notes/extending.func.html#automatically-generate-a-vmap-rule
    # should work since we only use PyTorch operations

    @staticmethod
    def forward(
        numbers: Tensor,
        weights: Tensor,
        reference: Reference,
        chunk_size: None | int = None,
    ) -> Tensor:
        if chunk_size is None:
            return _atomic_c6_full(numbers, weights, reference)

        return _atomic_c6_chunked(numbers, weights, reference, chunk_size)

    @staticmethod
    def setup_context(
        ctx: CTX,
        inputs: tuple[Tensor, Tensor, Reference, int | None],
        output: Tensor,
    ) -> None:
        numbers, weights, reference, chunk_size = inputs

        ctx.save_for_backward(numbers, weights)
        ctx.chunk_size = chunk_size
        ctx.reference = reference
