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
Utility functions: Gradient
===========================

Utilities for calculating gradients and Hessians.
"""
import torch

from ..__version__ import __torch_version__
from ..typing import Any, Callable, Tensor, Tuple

if __torch_version__ < (2, 0, 0):  # type: ignore, pragma: no cover
    try:
        from functorch import jacrev  # type: ignore
    except ModuleNotFoundError:
        jacrev = None
        from torch.autograd.functional import jacobian  # type: ignore

else:  # pragma: no cover
    from torch.func import jacrev  # type: ignore


def jac(f: Callable[..., Tensor], argnums: int = 0) -> Any:
    """
    Wrapper for Jacobian calcluation.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.
    """

    if jacrev is None:  # pragma: no cover

        def wrap(*inps: Tuple[Any, ...]) -> Any:
            """
            Wrapper to imitate the calling signature of functorch's `jacrev`
            with `torch.autograd.functional.jacobian`.

            Parameters
            ----------
            inps : tuple[Any, ...]
                The input parameters of the function `f`.

            Returns
            -------
            Any
                Jacobian function.

            Raises
            ------
            RuntimeError
                The parameter selected for differentiation (via `argnums`) is
                not a tensor.
            """
            diffarg = inps[argnums]
            if not isinstance(diffarg, Tensor):
                raise RuntimeError(
                    f"The {argnums}'th input parameter must be a tensor but is "
                    f"of type '{type(diffarg)}'."
                )

            before = inps[:argnums]
            after = inps[(argnums + 1) :]

            # `jacobian` only takes tensors, requiring another wrapper than
            # passes the non-tensor arguments to the function `f`
            def _f(arg: Tensor) -> Tensor:
                return f(*(*before, arg, *after))

            return jacobian(_f, inputs=diffarg)  # type: ignore, pylint: disable=used-before-assignment

        return wrap

    return jacrev(f, argnums=argnums)  # type: ignore


def hessian(
    f: Callable[..., Tensor],
    inputs: Tuple[Any, ...],
    argnums: int,
    is_batched: bool = False,
) -> Tensor:
    """
    Wrapper for Hessian. The Hessian is the Jacobian of the gradient.

    PyTorch, however, suggests calculating the Jacobian of the Jacobian, which
    does not yield the correct shape in this case.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    inputs : tuple[Any, ...]
        The input parameters of `f`.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Tensor
        The Hessian.

    Raises
    ------
    RuntimeError
        The parameter selected for differentiation (via `argnums`) is not a
        tensor.
    """

    def _grad(*inps: Tuple[Any, ...]) -> Tensor:
        e = f(*inps).sum()

        if not isinstance(inps[argnums], Tensor):
            raise RuntimeError(
                f"The {argnums}'th input parameter must be a tensor but is of "
                f"type '{type(inps[argnums])}'."
            )

        # catch missing gradients
        if e.grad_fn is None:
            return torch.zeros_like(inps[argnums])  # type: ignore

        (g,) = torch.autograd.grad(
            e,
            inps[argnums],
            create_graph=True,
        )
        return g

    _jac = jac(_grad, argnums=argnums)

    if is_batched:
        raise NotImplementedError("Batched Hessian not available.")
        # dims = Tuple(None if x != argnums else 0 for x in range(len(inputs)))
        # _jac = torch.func.vmap(_jac, in_dims=dims)

    return _jac(*inputs)  # type: ignore
