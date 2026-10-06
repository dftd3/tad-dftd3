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
Test the rational damping parameters.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.typing import DD as DDict

from tad_dftd3 import (
    D3Model,
    Damping,
    DampingParam,
    RationalTwoBody,
    ZeroThreeBodyD3,
    ZeroTwoBody,
    dftd3,
    disp,
)
from tad_dftd3.damping import as_damping_param
from tad_dftd3.param import get_functional_params

from ..conftest import DEVICE

DD: DDict = {"dtype": torch.double, "device": DEVICE}


def _structure() -> Structure:
    return Structure(
        numbers=torch.tensor([8, 1, 1], device=DEVICE),
        positions=torch.tensor(
            [[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.0, -1.4, -0.9]], **DD
        ),
    )


def test_from_functional() -> None:
    param = DampingParam.from_functional("pbe", **DD)
    ref = get_functional_params("pbe", damping="bj", **DD)
    for key, value in ref.items():
        assert getattr(param, key) == value
    assert param.s9 == 1.0


def test_from_functional_no_atm() -> None:
    param = DampingParam.from_functional("pbe", atm=False, **DD)
    assert param.s9 is None


def test_object_equals_dict() -> None:
    structure = _structure()
    ref = get_functional_params("pbe", damping="bj", **DD)
    param = DampingParam.from_functional("pbe", **DD)
    assert torch.equal(dftd3(structure, ref), dftd3(structure, param))


def test_dict_sets_only_its_keys() -> None:
    """A dictionary is unpacked as it is: nothing is filled in."""
    param = as_damping_param({"a1": 0.5, "s8": 1.0})
    assert param.s9 is None and param.a2 is None


def test_missing_fields_raise() -> None:
    """A parameter the damping needs is not made up."""
    with pytest.raises(ValueError, match="requires.*'s8'"):
        dftd3(_structure(), DampingParam(a1=0.5, a2=5.0))
    with pytest.raises(ValueError, match="requires.*'a2'"):
        dftd3(_structure(), {"a1": 0.5, "s8": 1.0})


def test_unknown_and_zero_damping() -> None:
    with pytest.raises(TypeError, match="foo"):
        dftd3(_structure(), {"a1": 0.5, "foo": 1.0})

    # Zero damping parameters have no `a1` and `a2`, which are not made up.
    zero = get_functional_params("pbe", damping="zero")
    with pytest.raises(ValueError, match="requires.*a1"):
        dftd3(_structure(), zero)


def test_damping_combinations() -> None:
    """The two- and three-body damping are chosen independently."""
    structure = _structure()
    param = DampingParam.from_functional("pbe", **DD)

    both = Damping(RationalTwoBody(), ZeroThreeBodyD3())
    only_two = Damping(RationalTwoBody())
    default = dftd3(structure, param)
    assert torch.equal(dftd3(structure, param, damping=both), default)

    # `s9` is set, but there is no three-body damping
    no_atm = dftd3(structure, param.replace(s9=None))
    assert torch.equal(dftd3(structure, param, damping=only_two), no_atm)
    assert not torch.equal(no_atm, default)


def test_three_body_needs_s9() -> None:
    param = DampingParam.from_functional("pbe", atm=False, **DD)
    damping = Damping(RationalTwoBody(), ZeroThreeBodyD3())
    with pytest.raises(ValueError, match="requires.*s9"):
        dftd3(_structure(), param, damping=damping)


@pytest.mark.parametrize(
    "s9, three_body",
    [
        (None, False),
        (0.0, False),
        (0, False),
        (1.0, True),
        (torch.tensor(0.0), True),
        (torch.tensor(1.0), True),
    ],
)
def test_default_damping_three_body(s9: object, three_body: bool) -> None:
    """
    Whether the default damping has the three-body term is a static choice:
    a Python number decides by its value, a tensor always has the term.
    """
    param = DampingParam(s8=1.0, a1=0.4, a2=5.0, s9=s9)  # type: ignore[arg-type]
    assert (disp.default_damping(param).three is not None) is three_body


def test_default_damping_named() -> None:
    """The default two-body damping is the one the parameters name."""
    assert isinstance(disp.default_damping(DampingParam()).two, RationalTwoBody)
    zero = DampingParam(damping="zero")
    assert isinstance(disp.default_damping(zero).two, ZeroTwoBody)


def test_tensor_s9_zero_derivative() -> None:
    """
    A tensor `s9` of zero still gets the three-body term, whose derivative
    with respect to `s9` is the three-body energy at ``s9 = 1``.
    """
    structure = _structure()
    param = DampingParam.from_functional("pbe", **DD)

    s9 = torch.tensor(0.0, **DD, requires_grad=True)
    energy = dftd3(structure, param.replace(s9=s9)).sum()
    (grad,) = torch.autograd.grad(energy, s9)

    c6 = D3Model().c6(structure)
    e3 = disp.dispersion3(structure, param.replace(s9=1.0), c6).sum()
    assert e3 != 0.0
    assert pytest.approx(e3.item(), abs=1e-14, rel=1e-12) == grad.item()

    # no gradient needed: the term is there, and zero
    no_grad = dftd3(structure, param.replace(s9=torch.tensor(0.0, **DD)))
    no_atm = dftd3(structure, param.replace(s9=None))
    assert pytest.approx(no_atm.cpu(), abs=1e-15) == no_grad.cpu()


def test_vmap_over_s8() -> None:
    structure = _structure()
    param = DampingParam.from_functional("pbe", atm=False, **DD)
    s8 = torch.tensor([0.5, 0.9], **DD)

    batched = torch.func.vmap(
        lambda s: dftd3(structure, param.replace(s8=s)).sum()
    )(s8)
    looped = torch.stack(
        [dftd3(structure, param.replace(s8=s)).sum() for s in s8]
    )
    assert pytest.approx(looped.cpu()) == batched.cpu()


def test_to_dtype() -> None:
    param = DampingParam.from_functional("pbe", **DD)
    a1 = param.to(dtype=torch.float32).a1
    assert isinstance(a1, torch.Tensor) and a1.dtype == torch.float32
