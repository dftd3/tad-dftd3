r"""
Rational (Becke-Johnson) damping function
=========================================

This module defines the rational damping function, also known as Becke-Johnson
damping.

.. math::

    f^n_{\text{damp}}\left(R_0^{\text{AB}}\right) =
    \dfrac{R^n_{\text{AB}}}{R^n_{\text{AB}} +
    \left( a_1 R_0^{\text{AB}} + a_2 \right)^n}
"""
import torch

from .. import defaults
from ..typing import Dict, Tensor


def rational_damping(
    order: int,
    distances: Tensor,
    qq: Tensor,
    param: Dict[str, Tensor],
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
