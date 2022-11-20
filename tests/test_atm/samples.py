"""
Data for testing D4 coordination number (taken from D4 testsuite).
"""
from __future__ import annotations

import torch

from tad_dftd3.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference values."""

    energy: Tensor
    """DFT-D3(BJ) dispersion energy without ATM term."""

    energy_atm: Tensor
    """DFT-D3(BJ)-ATM dispersion energy."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values."""


refs: dict[str, Refs] = {
    "SiH4": {
        "energy": torch.tensor(
            [
                -9.2481575005393872e-004,
                -3.6494949521315417e-004,
                -3.6494949521315417e-004,
                -3.6494949521315417e-004,
                -3.6494949521315417e-004,
            ],
            dtype=torch.float64,
        ),
        "energy_atm": torch.tensor(
            [
                -9.2481446570860746e-004,
                -3.6487414688948653e-004,
                -3.6487414688948653e-004,
                -3.6487414688948653e-004,
                -3.6487414688948653e-004,
            ],
            dtype=torch.float64,
        ),
    },
    "MB16_43_01": {
        "energy": torch.tensor(
            [
                -2.8788632548321321e-003,
                -6.3435979775151754e-004,
                -9.6167619562274962e-004,
                -7.9723260613915258e-004,
                -7.9238263177385578e-004,
                -7.4485995467369389e-004,
                -1.0311812354479540e-003,
                -1.0804678845482093e-003,
                -2.1424517331896948e-003,
                -5.3905710617330410e-004,
                -7.3549132878459982e-004,
                -2.9718856310496566e-003,
                -1.9053629060228276e-003,
                -1.8362475794413465e-003,
                -1.7182276597931356e-003,
                -4.2417715940356341e-003,
            ],
            dtype=torch.float64,
        ),
        "energy_atm": torch.tensor(
            [
                -2.8718125389999259e-003,
                -6.3328090446635918e-004,
                -9.5663711211542641e-004,
                -7.9370460692262154e-004,
                -7.9033697856002835e-004,
                -7.4037167668508294e-004,
                -1.0277787758263043e-003,
                -1.0733979636313967e-003,
                -2.1410728848939844e-003,
                -5.3484648487498051e-004,
                -7.3184554571681479e-004,
                -2.9622995709883419e-003,
                -1.9025657858451914e-003,
                -1.8324762672052280e-003,
                -1.7135582283322110e-003,
                -4.2406598201600847e-003,
            ],
            dtype=torch.float64,
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
