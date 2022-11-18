"""
Data for testing D4 coordination number (taken from D4 testsuite).
"""

import torch

from tad_dftd3.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference values."""

    energy: Tensor
    """DFT-D4 coordination number"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values."""


refs: dict[str, Refs] = {
    "SiH4": {
        "energy": torch.tensor(
            [
                -9.2481446570860746e-004,
                -3.6487414688948653e-004,
                -3.6487414688948653e-004,
                -3.6487414688948653e-004,
                -3.6487414688948653e-004,
            ]
        )
    },
    "MB16_43_01": {
        "energy": torch.tensor(
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
            ]
        )
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
