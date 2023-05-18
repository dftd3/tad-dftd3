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
Torch autodiff DFT-D3
=====================

Implementation of the DFT-D3 dispersion model in PyTorch.
This module allows to process a single structure or a batch of structures for
the calculation of atom-resolved dispersion energies.

.. note::

   This project is still in early development and the API is subject to change.
   Contributions are welcome, please checkout our
   `contributing guidelines <https://github.com/dftd3/tad-dftd3/blob/main/CONTRIBUTING.md>`_.


Example
-------
>>> import torch
>>> import tad_dftd3 as d3
>>> numbers = d3.util.pack((  # S22 system 4: formamide dimer
...     d3.util.to_number("C C N N H H H H H H O O".split()),
...     d3.util.to_number("C O N H H H".split()),
... ))
>>> positions = d3.util.pack((
...     torch.tensor([  # coordinates in Bohr
...         [-3.81469488143921, +0.09993441402912, 0.00000000000000],
...         [+3.81469488143921, -0.09993441402912, 0.00000000000000],
...         [-2.66030049324036, -2.15898251533508, 0.00000000000000],
...         [+2.66030049324036, +2.15898251533508, 0.00000000000000],
...         [-0.73178529739380, -2.28237795829773, 0.00000000000000],
...         [-5.89039325714111, -0.02589114569128, 0.00000000000000],
...         [-3.71254944801331, -3.73605775833130, 0.00000000000000],
...         [+3.71254944801331, +3.73605775833130, 0.00000000000000],
...         [+0.73178529739380, +2.28237795829773, 0.00000000000000],
...         [+5.89039325714111, +0.02589114569128, 0.00000000000000],
...         [-2.74426102638245, +2.16115570068359, 0.00000000000000],
...         [+2.74426102638245, -2.16115570068359, 0.00000000000000],
...     ]),
...     torch.tensor([
...         [-0.55569743203406, +1.09030425468557, 0.00000000000000],
...         [+0.51473634678469, +3.15152550263611, 0.00000000000000],
...         [+0.59869690244446, -1.16861263789477, 0.00000000000000],
...         [-0.45355203669134, -2.74568780438064, 0.00000000000000],
...         [+2.52721209544999, -1.29200800956867, 0.00000000000000],
...         [-2.63139587595376, +0.96447869452240, 0.00000000000000],
...     ]),
... ))
>>> param = dict( # ωB97M-D3(BJ) parameters
...     a1=torch.tensor(0.5660),
...     s8=torch.tensor(0.3908),
...     a2=torch.tensor(3.1280),
... )
>>> energy = torch.sum(d3.dftd3(numbers, positions, param), -1)
>>> torch.set_printoptions(precision=7)
>>> print(energy)  # Energies in Hartree
tensor([-0.0124292, -0.0045002])
>>> print(energy[0] - 2*energy[1])
tensor(-0.0034288)
"""
import torch

from . import damping, data, disp, model, ncoord, reference, util
from .typing import (
    CountingFunction,
    DampingFunction,
    Dict,
    Optional,
    Tensor,
    WeightingFunction,
    DD,
)


def dftd3(
    numbers: Tensor,
    positions: Tensor,
    param: Dict[str, Tensor],
    *,
    ref: Optional[reference.Reference] = None,
    rcov: Optional[Tensor] = None,
    rvdw: Optional[Tensor] = None,
    r4r2: Optional[Tensor] = None,
    cutoff: Optional[Tensor] = None,
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = model.gaussian_weight,
    damping_function: DampingFunction = damping.rational_damping,
) -> Tensor:
    """
    Evaluate DFT-D3 dispersion energy for a batch of geometries.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers of the atoms in the system.
    positions : torch.Tensor
        Cartesian coordinates of the atoms in the system.
    param : dict[str, Tensor]
        DFT-D3 damping parameters.
    ref : reference.Reference, optional
        Reference C6 coefficients.
    rcov : torch.Tensor, optional
        Covalent radii of the atoms in the system.
    rvdw : torch.Tensor, optional
        Van der Waals radii of the atoms in the system.
    r4r2 : torch.Tensor, optional
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : Callable, optional
        Damping function evaluate distance dependent contributions.
    weighting_function : Callable, optional
        Function to calculate weight of individual reference systems.
    counting_function : Callable, optional
        Calculates counting value in range 0 to 1 for each atom pair.

    Returns
    -------
    torch.Tensor
        DFT-D3 dispersion energy for each geometry.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = torch.tensor(50.0, **dd)
    if ref is None:
        ref = reference.Reference(**dd)
    if rcov is None:
        rcov = data.covalent_rad_d3[numbers].type(positions.dtype).to(positions.device)
    if rvdw is None:
        rvdw = (
            data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
            .type(positions.dtype)
            .to(positions.device)
        )
    if r4r2 is None:
        r4r2 = (
            data.sqrt_z_r4_over_r2[numbers].type(positions.dtype).to(positions.device)
        )

    cn = ncoord.coordination_number(numbers, positions, rcov, counting_function)
    weights = model.weight_references(numbers, cn, ref, weighting_function)
    c6 = model.atomic_c6(numbers, weights, ref)
    energy = disp.dispersion(
        numbers,
        positions,
        param,
        c6,
        rvdw,
        r4r2,
        damping_function,
        cutoff=cutoff,
    )

    return energy
