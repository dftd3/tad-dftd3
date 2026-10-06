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
"""
The damping variants against s-dftd3.

Rational (BJ and its refit BJM, which has the same form), zero, modified
zero, optimized power, CSO and Z damping of the two-body term, with the
zero-damped three-body term. Energies and gradients of a molecule, a padded
batch and a cell are compared to s-dftd3, the sparse evaluation to the dense
one, and the energy is run under the transforms of ``torch``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.batch import pack
from tad_mctc.io.structure import pack_structures
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import Cutoff, Damping, DampingParam, dftd3
from tad_dftd3.damping import (
    CSOTwoBody,
    ModifiedZeroTwoBody,
    OptimizedPowerTwoBody,
    RationalTwoBody,
    ZeroThreeBodyD3,
    ZeroTwoBody,
    ZTwoBody,
    damping_from_name,
)

from ..cells import cells
from ..conftest import DEVICE, compile_test
from ..reference import reference_energy_per_atom, reference_gradient
from ..utils import load_structure

DD64: DD = {"device": DEVICE, "dtype": torch.double}

# variant of the data base -> (functional, name of the damping)
VARIANTS = {
    "bj": ("blyp", "rational"),
    "bjm": ("blyp", "rational"),
    "zero": ("blyp", "zero"),
    "zerom": ("blyp", "mzero"),
    "op": ("blyp", "optimizedpower"),
    "cso": ("blyp", "cso"),
}
CLASSES = {
    "rational": RationalTwoBody,
    "zero": ZeroTwoBody,
    "mzero": ModifiedZeroTwoBody,
    "optimizedpower": OptimizedPowerTwoBody,
    "cso": CSOTwoBody,
    "z": ZTwoBody,
}
NAMES = [*VARIANTS, "z"]

# short enough to cut images of a cell
cutoff = Cutoff(cn=15.0, disp2=20.0, disp3=15.0)


def make_param(variant: str, dd: DD = DD64) -> DampingParam:
    if variant == "z":
        return DampingParam(
            a1=torch.tensor(1.8, **dd),
            s6=torch.tensor(1.0, **dd),
            s8=torch.tensor(0.7, **dd),
            s9=torch.tensor(1.0, **dd),
            damping="z",
        )
    functional, _ = VARIANTS[variant]
    return DampingParam.from_functional(functional, damping=variant, **dd)


def as_reference(param: DampingParam) -> dict[str, Any]:
    """The parameters as the mapping `test/reference.py` takes."""
    out: dict[str, Any] = {
        name: getattr(param, name)
        for name in param._child_names  # pylint: disable=protected-access
        if getattr(param, name) is not None
    }
    out["damping"] = param.damping
    return out


def alternate(structure: Structure) -> Structure:
    """
    The atoms alternately H and He. s-dftd3 evaluates the Z damping with the
    index of the species (in order of first appearance) in place of the
    atomic number, which are the same here.
    """
    index = torch.arange(structure.numbers.shape[-1], device=DEVICE)
    numbers = torch.where(index % 2 == 0, 1, 2)
    return structure.replace(numbers=numbers)


def molecule(variant: str = "") -> Structure:
    structure = load_structure("heavy28", "pbh4_bih3", DD64)
    return alternate(structure) if variant == "z" else structure


def cell(variant: str = "") -> Structure:
    return alternate(cells["urea"]) if variant == "z" else cells["urea"]


def singles(variant: str = "") -> list[Structure]:
    out = [
        load_structure("mb16_43", "LiH", DD64),
        load_structure("mb16_43", "SiH4", DD64),
    ]
    return [alternate(s) for s in out] if variant == "z" else out


def batch(variant: str = "") -> Structure:
    return pack_structures(singles(variant))


@pytest.mark.parametrize("variant", NAMES)
def test_selected_by_the_parameters(variant: str) -> None:
    """The damping is the one the parameters name, if none is given."""
    param = make_param(variant)
    name = param.damping
    assert name is not None
    assert isinstance(damping_from_name(name).two, CLASSES[name])

    structure = molecule()
    explicit = Damping(CLASSES[name](), ZeroThreeBodyD3())
    assert torch.equal(
        dftd3(structure, param), dftd3(structure, param, damping=explicit)
    )


@pytest.mark.parametrize("variant", NAMES)
@pytest.mark.parametrize("kind", ["molecule", "batch", "cell"])
def test_energy(variant: str, kind: str) -> None:
    param = make_param(variant)
    ref_param = as_reference(param)

    if kind == "batch":
        structure = batch(variant)
        energy = dftd3(structure, param)
        ref = pack(
            [reference_energy_per_atom(s, ref_param) for s in singles(variant)]
        )
        # padding atoms have no energy
        assert (energy[structure.numbers == 0] == 0).all()
    else:
        structure = molecule(variant) if kind == "molecule" else cell(variant)
        cut = None if kind == "molecule" else cutoff
        energy = dftd3(structure, param, cutoff=cut or Cutoff())
        ref = reference_energy_per_atom(structure, ref_param, cutoff=cut)

    assert pytest.approx(ref.cpu(), abs=1e-10, rel=0) == energy.detach().cpu()


@pytest.mark.parametrize("variant", NAMES)
@pytest.mark.parametrize("kind", ["molecule", "cell"])
def test_gradient(variant: str, kind: str) -> None:
    param = make_param(variant)
    structure = molecule(variant) if kind == "molecule" else cell(variant)
    cut = None if kind == "molecule" else cutoff

    positions = structure.positions.clone().requires_grad_(True)
    energy = dftd3(
        structure.replace(positions=positions),
        param,
        cutoff=cut or Cutoff(),
    )
    (grad,) = torch.autograd.grad(energy.sum(), positions)

    ref, _ = reference_gradient(structure, as_reference(param), cutoff=cut)
    assert pytest.approx(ref.cpu(), abs=1e-9, rel=0) == grad.cpu()


@pytest.mark.parametrize("variant", NAMES)
def test_padded_gradient_is_finite(variant: str) -> None:
    """Padding atoms (no radius, no atomic number) give no NaN."""
    param = make_param(variant)
    structure = batch()
    positions = structure.positions.clone().requires_grad_(True)
    energy = dftd3(structure.replace(positions=positions), param)
    (grad,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("variant", NAMES)
@pytest.mark.parametrize("kind", ["molecule", "batch", "cell"])
def test_sparse_matches_dense(variant: str, kind: str) -> None:
    param = make_param(variant)
    structure = {
        "molecule": molecule(),
        "batch": batch(),
        "cell": cells["urea"],
    }[kind]
    cut = Cutoff() if kind != "cell" else cutoff

    dense = dftd3(structure, param, cutoff=cut)
    sparse = dftd3(structure, param, cutoff=cut, sparse=True)
    assert pytest.approx(dense.cpu(), abs=1e-10, rel=0) == sparse.cpu()


@pytest.mark.parametrize("variant", NAMES)
def test_float32_gradient(variant: str) -> None:
    """No overflow (CSO has an exponential) in single precision."""
    dd: DD = {"device": DEVICE, "dtype": torch.float}
    param = make_param(variant, dd)
    structure = molecule().to(**dd)
    positions = structure.positions.clone().requires_grad_(True)
    energy = dftd3(structure.replace(positions=positions), param)
    (grad,) = torch.autograd.grad(energy.sum(), positions)
    assert torch.isfinite(energy).all() and torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "variant, needed",
    [
        ("zero", "rs6"),
        ("zerom", "bet"),
        ("op", "bet"),
        ("cso", "a1"),
        ("z", "a1"),
    ],
)
def test_missing_parameters(variant: str, needed: str) -> None:
    """The parameters a variant needs are not made up."""
    param = make_param(variant)
    with pytest.raises(ValueError, match=f"requires.*'{needed}'"):
        dftd3(molecule(), param.replace(**{needed: None}))


def test_other_damping_than_named() -> None:
    """A damping that is given wins over the name in the parameters."""
    param = make_param("zero")
    with pytest.raises(ValueError, match="Rational damping requires"):
        dftd3(molecule(), param, damping=Damping(RationalTwoBody()))
    with pytest.raises(ValueError, match="Unknown damping"):
        damping_from_name("nope")


@pytest.mark.parametrize("variant", NAMES)
def test_vmap(variant: str) -> None:
    param = make_param(variant)
    structure = batch()
    pos = structure.positions

    def energy(n: Tensor, p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=n, positions=p), param)

    batched = torch.func.vmap(energy)(structure.numbers, pos)
    looped = dftd3(structure, param)
    assert pytest.approx(looped.cpu(), abs=1e-10, rel=0) == batched.cpu()


@pytest.mark.parametrize("variant", NAMES)
def test_jacrev_jacfwd(variant: str) -> None:
    param = make_param(variant)
    structure = load_structure("mb16_43", "SiH4", DD64)

    def energy(p: Tensor) -> Tensor:
        return dftd3(
            Structure(numbers=structure.numbers, positions=p), param
        ).sum()

    rev = torch.func.jacrev(energy)(structure.positions)
    fwd = torch.func.jacfwd(energy)(structure.positions)
    assert pytest.approx(rev.cpu(), abs=1e-10, rel=0) == fwd.cpu()


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
@pytest.mark.parametrize("variant", NAMES)
def test_compile_fullgraph(variant: str) -> None:
    param = make_param(variant)
    structure = load_structure("mb16_43", "SiH4", DD64)

    def energy(p: Tensor) -> Tensor:
        return dftd3(Structure(numbers=structure.numbers, positions=p), param)

    ref = energy(structure.positions)
    out = compile_test(energy, fullgraph=True)(structure.positions)
    assert pytest.approx(ref.cpu(), abs=1e-10, rel=0) == out.cpu()
