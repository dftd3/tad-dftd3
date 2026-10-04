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
Testing dispersion Hessian (autodiff).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Callable, Tensor
from torch.func import hessian, jacrev, vmap

from tad_dftd3.disp import dftd3

from ..conftest import DEVICE
from ..reference import reference_hessian
from ..utils import load_sample, load_structure

sample_list: list[tuple[str, str]] = [
    ("mb16_43", "LiH"),
    ("mb16_43", "SiH4"),
    ("heavy28", "pbh4_bih3"),
    ("mb16_43", "01"),
]

tol = 1e-8


def _hessian(mode: str, f: Callable[..., Tensor]) -> Callable[..., Tensor]:
    """
    Hessian of `f` with respect to its second argument (the positions):
    reverse-over-reverse or forward-over-reverse (`torch.func.hessian`).
    """
    if mode == "rev":
        return jacrev(jacrev(f, argnums=1), argnums=1)
    return hessian(f, argnums=1)


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source", sample_list)
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_single(dtype: torch.dtype, source: tuple[str, str], mode: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers, positions = load_sample(*source, dd)

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref = reference_hessian(load_structure(*source, dd), param)

    def _energy(numbers: Tensor, positions: Tensor) -> Tensor:
        """
        Closure over non-tensor argument `param` for `dftd3` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd3(
            Structure(numbers=numbers, positions=positions), param
        ).sum(-1)

    hess = _hessian(mode, _energy)(numbers, positions)
    assert isinstance(hess, Tensor)

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("source1", [("mb16_43", "LiH")])
@pytest.mark.parametrize("source2", sample_list)
def test_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack([load_sample(*src, dd)[0] for src in (source1, source2)])
    positions = pack([load_sample(*src, dd)[1] for src in (source1, source2)])

    # GFN1-xTB parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(2.40000000, **dd),
        "s9": torch.tensor(0.00000000, **dd),
        "a1": torch.tensor(0.63000000, **dd),
        "a2": torch.tensor(5.00000000, **dd),
    }

    ref = pack(
        [
            reference_hessian(load_structure(*src, dd), param)
            for src in (source1, source2)
        ]
    )

    def _energy(numbers: Tensor, positions: Tensor) -> Tensor:
        """
        Closure over non-tensor argument `param` for `dftd3` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd3(
            Structure(numbers=numbers, positions=positions), param
        ).sum(-1)

    hess = vmap(_hessian("rev", _energy), in_dims=(0, 0))(numbers, positions)
    assert isinstance(hess, Tensor)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()
