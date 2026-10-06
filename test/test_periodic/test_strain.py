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
Strain derivative of the periodic dispersion energy.

A homogeneous strain ``eps`` deforms positions and lattice vectors (rows)
alike, ``x -> x (1 + eps)``. The derivative of the energy with respect to it
at ``eps = 0`` is the virial that s-dftd3 returns; here it is taken by
autograd through the strain itself, rather than assembled from the position
and lattice gradients as in ``test_grad.py``, for the dense evaluation and for
the one over a pre-built neighbour list, which stays valid under the strain.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.io.structure import pack_structures
from tad_mctc.neighbor import pair_distance_squared
from tad_mctc.neighbor.list import build_neighborlist, build_neighborlists
from tad_mctc.typing import DD, Tensor

from tad_dftd3 import dftd3
from tad_dftd3.cutoff import Cutoff

from ..cells import TRICLINIC, cells, param_on, random_cell
from ..conftest import DEVICE
from ..reference import reference_gradient

cutoff = Cutoff(cn=15.0, disp2=20.0)

tol = 1e-12

DD64: DD = {"device": DEVICE, "dtype": torch.double}


def _strained(structure: Structure, strain: Tensor) -> Structure:
    """`structure` deformed by `strain`, ``(..., 3, 3)``: ``x -> x (1 + eps)``."""
    assert structure.lattice is not None
    return structure.replace(
        positions=structure.positions + structure.positions @ strain,
        lattice=structure.lattice + structure.lattice @ strain,
    )


def _energy(
    structure: Structure, strain: Tensor, sparse: bool, **kwargs: object
) -> Tensor:
    """Total energy of the strained `structure`, per system."""
    energy = dftd3(
        _strained(structure, strain),
        param_on(DD64),
        cutoff=cutoff,
        sparse=sparse,
        **kwargs,  # type: ignore[arg-type]
    )
    return energy.sum(-1)


def _virial(structure: Structure, sparse: bool) -> Tensor:
    """``dE/d(strain)`` at zero strain, by autograd, ``(..., 3, 3)``."""
    strain = torch.zeros(
        structure.positions.shape[:-2] + (3, 3), requires_grad=True, **DD64
    )
    energy = _energy(structure, strain, sparse)
    (virial,) = torch.autograd.grad(energy.sum(), strain)
    return virial


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("name", list(cells))
def test_virial_matches_reference(name: str, sparse: bool) -> None:
    structure = cells[name].to(DEVICE)

    _, ref_virial = reference_gradient(structure, param_on(DD64), cutoff=cutoff)
    virial = _virial(structure, sparse)

    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "periodic", [[True, True, False], [True, False, False]]
)
def test_virial_low_dimensional(periodic: list[bool], sparse: bool) -> None:
    # A slab and a chain: the strain acts on the positions along the open
    # axes as well, but moves no image there.
    mask = torch.tensor(periodic, device=DEVICE)
    structure = random_cell(TRICLINIC, 5, DD64, seed=11, periodic=mask)

    _, ref_virial = reference_gradient(structure, param_on(DD64), cutoff=cutoff)
    virial = _virial(structure, sparse)

    assert pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial.cpu()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_virial_is_symmetric(name: str, sparse: bool) -> None:
    """
    The energy is invariant under rotations, so its derivative with respect to
    the antisymmetric part of the strain (an infinitesimal rotation) vanishes.
    """
    virial = _virial(cells[name].to(DEVICE), sparse)

    assert pytest.approx(virial.cpu(), abs=tol, rel=0) == virial.mT.cpu()


def _distance_to_cutoffs(structure: Structure) -> float:
    """
    Smallest distance of any pair (or pair with an image) from either hard
    cutoff, in Bohr.
    """
    assert structure.lattice is not None
    nbl = build_neighborlist(structure, cutoff.disp2 + 1.0)
    real = nbl.mask
    distances = torch.sqrt(
        pair_distance_squared(
            nbl.idx_i[real],
            nbl.idx_j[real],
            nbl.shift[real],
            structure.positions,
            shared_lattice=structure.lattice,
            system_lattices=None,
            atoms_per_system=structure.positions.shape[-2],
        )
    )
    return min(
        float((distances - cut).abs().min())
        for cut in (cutoff.cn, cutoff.disp2)
    )


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_virial_finite_difference(name: str) -> None:
    """Central differences of the energy, for each strain component."""
    structure = cells[name].to(DEVICE)

    # The cutoffs are hard, so the energy jumps where a pair crosses one,
    # which central differences would pick up and the analytic derivative
    # does not. Only meaningful if no pair is within reach of a cutoff.
    step = 1e-5
    margin = 100 * step
    nearest = _distance_to_cutoffs(structure)
    if nearest < margin:
        pytest.skip(f"a pair is {nearest:.1e} Bohr from a hard cutoff")

    virial = _virial(structure, sparse=False)

    for i in range(3):
        for j in range(3):
            strain = torch.zeros(3, 3, **DD64)
            strain[i, j] = step
            plus = _energy(structure, strain, sparse=False)
            minus = _energy(structure, -strain, sparse=False)

            numerical = (plus - minus) / (2 * step)
            assert pytest.approx(virial[i, j].item(), abs=1e-10) == (
                numerical.item()
            )


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_list_with_skin_under_finite_strain(name: str) -> None:
    """
    A list built once, for the unstrained cell, as for the steps of a cell
    optimisation: its integer shifts still index the right images of the
    strained cell, and the skin holds the pairs that the strain moves into
    the cutoff. Without a skin those pairs would be missing, which is what
    the skin is for.
    """
    structure = cells[name].to(DEVICE)
    nbl_cn, nbl_disp2 = build_neighborlists(
        structure, (cutoff.cn, cutoff.disp2), skin=1.0
    )

    gen = torch.Generator().manual_seed(3)
    strain = 5e-3 * torch.randn(
        3, 3, generator=gen, dtype=torch.double, device="cpu"
    )
    strain = strain.to(DEVICE)

    dense = _energy(structure, strain, sparse=False)
    listed = _energy(
        structure, strain, sparse=False, nbl_cn=nbl_cn, nbl_disp2=nbl_disp2
    )

    assert pytest.approx(dense.cpu(), abs=1e-12, rel=0) == listed.cpu()


@pytest.mark.parametrize("name", ["urea", "periodic_triclinic"])
def test_strain_hessian_sparse_matches_dense(name: str) -> None:
    """The second derivative with respect to the strain (elastic constants)."""
    structure = cells[name].to(DEVICE)

    def hessian(sparse: bool) -> Tensor:
        strain = torch.zeros(3, 3, requires_grad=True, **DD64)
        energy = _energy(structure, strain, sparse)
        (grad,) = torch.autograd.grad(energy, strain, create_graph=True)

        rows = [
            torch.autograd.grad(g, strain, retain_graph=True)[0]
            for g in grad.reshape(-1)
        ]
        return torch.stack(rows).reshape(3, 3, 3, 3)

    assert (
        pytest.approx(hessian(False).cpu(), abs=1e-11, rel=0)
        == hessian(True).cpu()
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_batch_virial_matches_single(sparse: bool) -> None:
    """Each cell of a padded batch gets its own strain and virial."""
    sources = [
        random_cell(TRICLINIC, 3, DD64, seed=1),
        random_cell(TRICLINIC * 1.3, 5, DD64, seed=2),
    ]
    batch = pack_structures(sources)

    virial = _virial(batch, sparse)

    assert virial.shape == (2, 3, 3)
    for i, single in enumerate(sources):
        _, ref_virial = reference_gradient(
            single, param_on(DD64), cutoff=cutoff
        )
        assert (
            pytest.approx(ref_virial.cpu(), abs=tol, rel=0) == virial[i].cpu()
        )
