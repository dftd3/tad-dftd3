"""
Three-body (Axilrod-Teller-Muto, ATM) dispersion term.
"""
import torch

from .. import defaults
from ..typing import Tensor
from ..util import real_pairs, real_triples


def dispersion_atm(
    numbers: Tensor,
    positions: Tensor,
    c6: Tensor,
    rvdw: Tensor,
    cutoff: Tensor,
    s9: Tensor = torch.tensor(defaults.S9),
    rs9: Tensor = torch.tensor(defaults.RS9),
    alp: Tensor = torch.tensor(defaults.ALP),
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    rvdw : Tensor
        Van der Waals radii of the atoms in the system.
    cutoff : Tensor
        Real-space cutoff.
    s9 : Tensor, optional
        Scaling for dispersion coefficients. Defaults to `1.0`.
    rs9 : Tensor, optional
        Scaling for van-der-Waals radii in damping function. Defaults to `4.0/3.0`.
    alp : Tensor, optional
        Exponent of zero damping function. Defaults to `14.0`.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.
    """
    s9 = s9.type(positions.dtype).to(positions.device)
    rs9 = rs9.type(positions.dtype).to(positions.device)
    alp = alp.type(positions.dtype).to(positions.device)

    cutoff2 = cutoff * cutoff
    srvdw = rs9 * rvdw

    # C9_ABC = s9 * sqrt(|C6_AB * C6_AC * C6_BC|)
    c9 = s9 * torch.sqrt(
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
            real_pairs(numbers, diagonal=False),
            torch.cdist(
                positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
            ),
            positions.new_tensor(torch.finfo(positions.dtype).eps),
        ),
        2.0,
    )

    r2ij = distances.unsqueeze(-1)
    r2ik = distances.unsqueeze(-2)
    r2jk = distances.unsqueeze(-3)
    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    fdamp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** ((alp + 2.0) / 3.0))

    s = (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik)
    ang = torch.where(
        real_triples(numbers, diagonal=False)
        * (r2ij <= cutoff2)
        * (r2jk <= cutoff2)
        * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        positions.new_tensor(0.0),
    )

    energy = ang * fdamp * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0
