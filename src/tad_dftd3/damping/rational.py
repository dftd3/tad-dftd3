"""
Rational (Becke-Johnson) damping function.
"""
from __future__ import annotations

import torch

from .. import defaults
from ..typing import Tensor

__all__ = ["rational_damping"]


def rational_damping(
    order: int,
    distances: Tensor,
    qq: Tensor,
    param: dict[str, Tensor],
) -> Tensor:
    """
    Rational damped dispersion interaction between pairs.

    Parameters
    ----------
    order : int
        Order of the dispersion interaction, e.g.
        6 for dipole-dipole, 8 for dipole-quadrupole and so on.
    distances : Tensor
        Pairwise distances between atoms in the system.
    qq : Tensor
        Quotient of C8 and C6 dispersion coefficients.
    param : dict[str, Tensor]
        DFT-D3 damping parameters.

    Returns
    -------
    Tensor
        Values of the damping function.
    """
    a1 = param.get("a1", distances.new_tensor(defaults.A1))
    a2 = param.get("a2", distances.new_tensor(defaults.A1))
    return 1.0 / (distances.pow(order) + (a1 * torch.sqrt(qq) + a2).pow(order))