"""
Version module for tad_dftd3.
"""
import torch

__version__ = "0.1.1"

__torch_version__ = tuple(
    int(x) for x in torch.__version__.split("+", maxsplit=1)[0].split(".")
)
