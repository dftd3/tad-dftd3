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
Testing `torch.compile` of the DFT-D3 energy.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tad_mctc import Structure
from tad_mctc._version import __tversion__
from tad_mctc.batch import pack
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import data, dftd3, disp
from tad_dftd3.cutoff import Cutoff

from ..conftest import DEVICE, compile_test, requires_compile
from ..utils import load_sample


@pytest.fixture(name="dispersion3_calls")
def fixture_dispersion3_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """
    Count the evaluations (or traces) of the three-body term. A plain
    function, since Dynamo cannot trace a `unittest.mock` object.
    """
    calls: list[int] = []
    original = disp.dispersion3

    def counting(*args: Any, **kwargs: Any) -> Tensor:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(disp, "dispersion3", counting)
    return calls


pytestmark = pytest.mark.usefixtures("reset_dynamo")


tol = 1e-8

names = [("mb16_43", "SiH4")]


def _setup(
    source: tuple[str, str], s9: float
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    dd: DD = {"device": DEVICE, "dtype": torch.double}

    numbers, positions = load_sample(*source, dd)

    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(0.78981345, **dd),
        "s9": torch.tensor(s9, **dd),
        "a1": torch.tensor(0.49484001, **dd),
        "a2": torch.tensor(5.73083694, **dd),
    }
    return numbers, positions, param


@requires_compile
@pytest.mark.parametrize("source", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_graph_breaks(source: tuple[str, str], s9: float) -> None:
    """
    Compilation with graph breaks allowed. While tracing, the three-body term
    is always evaluated (scaled by `s9`), which must not change the result.
    """
    numbers, positions, param = _setup(source, s9)

    ref = dftd3(Structure(numbers=numbers, positions=positions), param)
    out = compile_test(
        lambda n, p: dftd3(Structure(numbers=n, positions=p), param)
    )(numbers, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@requires_compile
@pytest.mark.parametrize("source", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_fullgraph(source: tuple[str, str], s9: float) -> None:
    """
    A single graph, from a cold start: a graph-breaking compile earlier in the
    same process (e.g. `test_graph_breaks`) runs the untraceable code eagerly
    and can leave state behind that lets a later `fullgraph=True` pass.
    """

    numbers, positions, param = _setup(source, s9)

    ref = dftd3(Structure(numbers=numbers, positions=positions), param)
    compiled = compile_test(
        lambda n, p: dftd3(Structure(numbers=n, positions=p), param),
        fullgraph=True,
    )
    out = compiled(numbers, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@requires_compile
@pytest.mark.skipif(
    __tversion__ < (2, 5, 0),
    reason="`torch.compile` of `torch.func` transforms needs PyTorch 2.5.0.",
)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_fullgraph_vmap_jacrev(s9: float) -> None:
    """
    Compiled gradients of a padded batch, with respect to the positions and
    a shared per-element table.
    """

    dd: DD = {"device": DEVICE, "dtype": torch.double}
    _, _, param = _setup(("mb16_43", "SiH4"), s9)

    sources = [("mb16_43", "LiH"), ("mb16_43", "SiH4")]
    numbers = pack([load_sample(*src, dd)[0] for src in sources])
    positions = pack([load_sample(*src, dd)[1] for src in sources])
    r4r2 = data.R4R2(**dd)

    def energy(n: Tensor, p: Tensor, t: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=n, positions=p), param, r4r2_table=t
        ).sum(-1)

    grad = torch.func.vmap(
        torch.func.jacrev(energy, argnums=(1, 2)), in_dims=(0, 0, None)
    )
    ref = grad(numbers, positions, r4r2)
    out = compile_test(grad, fullgraph=True)(numbers, positions, r4r2)

    for r, o in zip(ref, out):
        assert pytest.approx(r.cpu(), abs=tol) == o.cpu()


@requires_compile
@pytest.mark.parametrize("source", names)
@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_fullgraph_changed_cutoff(source: tuple[str, str], s9: float) -> None:
    """
    A single graph with cutoffs short enough to cut pairs and triples of
    this molecule: the cutoffs reach the coordination number and both
    dispersion terms.
    """

    numbers, positions, param = _setup(source, s9)
    cutoff = Cutoff(cn=4.0, disp2=6.0, disp3=6.0)

    structure = Structure(numbers=numbers, positions=positions)
    ref = dftd3(structure, param, cutoff=cutoff)
    compiled = compile_test(
        lambda n, p: dftd3(
            Structure(numbers=n, positions=p), param, cutoff=cutoff
        ),
        fullgraph=True,
    )
    out = compiled(numbers, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()

    # only meaningful if the cutoffs change the result
    default = dftd3(structure, param)
    assert not torch.allclose(ref, default, atol=1e-10, rtol=0)


@requires_compile
@pytest.mark.parametrize("source", names)
def test_fullgraph_skips_atm_for_float_s9(
    source: tuple[str, str], dispersion3_calls: list[int]
) -> None:
    """
    A Python number `s9 = 0.0` is a constant to the trace, so the three-body
    term is not evaluated under `fullgraph=True`, unlike a tensor `s9`.
    """
    numbers, positions, param = _setup(source, 0.0)
    ref = dftd3(Structure(numbers=numbers, positions=positions), param)

    param_float = {**param, "s9": 0.0}
    compiled = compile_test(
        lambda n, p: dftd3(Structure(numbers=n, positions=p), param_float),
        fullgraph=True,
    )
    out = compiled(numbers, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
    assert len(dispersion3_calls) == 0

    # a tensor `s9` of zero is traced, so the term is evaluated
    torch._dynamo.reset()  # pylint: disable=protected-access
    compiled = compile_test(
        lambda n, p: dftd3(Structure(numbers=n, positions=p), param),
        fullgraph=True,
    )
    out = compiled(numbers, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
    assert len(dispersion3_calls) > 0


@requires_compile
@pytest.mark.parametrize("source", names)
def test_fullgraph_float_s9(source: tuple[str, str]) -> None:
    """A Python number `s9 = 1.0` gives the same energy as a tensor."""
    numbers, positions, param = _setup(source, 1.0)
    structure = Structure(numbers=numbers, positions=positions)
    ref = dftd3(structure, param)

    param_float = {**param, "s9": 1.0}
    assert (
        pytest.approx(ref.cpu(), abs=tol) == dftd3(structure, param_float).cpu()
    )

    compiled = compile_test(
        lambda n, p: dftd3(Structure(numbers=n, positions=p), param_float),
        fullgraph=True,
    )
    out = compiled(numbers, positions)

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()


@requires_compile
@pytest.mark.parametrize("source", names)
@pytest.mark.parametrize("fullgraph", [False, True])
def test_float_s9_as_argument(
    source: tuple[str, str], fullgraph: bool, dispersion3_calls: list[int]
) -> None:
    """
    As in the README, the `Structure` and `param` with `s9 = 0.0` are
    arguments of the compiled function, not captured: the three-body term
    is still skipped.
    """
    numbers, positions, param = _setup(source, 0.0)
    structure = Structure(numbers=numbers, positions=positions)
    ref = dftd3(structure, param)

    compiled = compile_test(dftd3, fullgraph=fullgraph)
    out = compiled(structure, {**param, "s9": 0.0})

    assert pytest.approx(ref.cpu(), abs=tol) == out.cpu()
    assert len(dispersion3_calls) == 0
