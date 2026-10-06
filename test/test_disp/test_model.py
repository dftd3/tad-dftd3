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
Test the dispersion model as a value.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.data.radii import COV_D3
from tad_mctc.typing import DD as DDict

from tad_dftd3 import Cutoff, D3Model, DampingParam, dftd3
from tad_dftd3.data import R4R2

from ..conftest import DEVICE

DD: DDict = {"dtype": torch.double, "device": DEVICE}


def _structure() -> Structure:
    return Structure(
        numbers=torch.tensor([8, 1, 1], device=DEVICE),
        positions=torch.tensor(
            [[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]], **DD
        ),
    )


def _param() -> DampingParam:
    return DampingParam.from_functional("pbe", **DD)


def test_equals_function() -> None:
    structure, param = _structure(), _param()
    assert torch.equal(D3Model()(structure, param), dftd3(structure, param))


def test_replace_cutoff() -> None:
    structure, param = _structure(), _param()
    cutoff = Cutoff(disp2=2.0, disp3=2.0)
    model = D3Model().replace(cutoff=cutoff)
    assert torch.equal(
        model(structure, param), dftd3(structure, param, cutoff=cutoff)
    )
    assert not torch.equal(model(structure, param), dftd3(structure, param))


def test_c6_shape() -> None:
    structure = _structure()
    assert D3Model().c6(structure).shape == (3, 3)


def test_table_gradient_and_to() -> None:
    """A tensor table is a pytree leaf and differentiable."""
    structure, param = _structure(), _param()
    r4r2 = R4R2(**DD).clone().requires_grad_()
    energy = D3Model(r4r2_table=r4r2)(structure, param).sum()
    (grad,) = torch.autograd.grad(energy, r4r2)
    assert grad.shape == r4r2.shape and grad.abs().sum() > 0

    model = D3Model(rcov_table=COV_D3(**DD))
    assert model.to(dtype=torch.float32).rcov_table.dtype == torch.float32  # type: ignore


def test_vmap_over_table() -> None:
    structure, param = _structure(), _param()
    base = R4R2(**DD)
    tables = torch.stack([base, 1.1 * base])
    batched = torch.func.vmap(
        lambda t: D3Model(r4r2_table=t)(structure, param).sum()
    )(tables)
    looped = torch.stack(
        [D3Model(r4r2_table=t)(structure, param).sum() for t in tables]
    )
    assert pytest.approx(looped.cpu()) == batched.cpu()
