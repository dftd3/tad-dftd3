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
Model: Dispersion model
=======================

Implementation of D3 model to obtain atomic C6 coefficients for a given
geometry.

Examples
--------
>>> import torch
>>> import tad_dftd3 as d3
>>> import tad_mctc as mctc
>>> numbers = mctc.convert.symbol_to_number(["O", "H", "H"])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
... ], dtype=torch.double)
>>> ref = d3.reference.Reference(dtype=torch.double)
>>> structure = mctc.Structure(numbers=numbers, positions=positions)
>>> cn_model = d3.ncoord.cn_d3.replace(cutoff=d3.defaults.D3_CN_CUTOFF)
>>> cn = cn_model(structure)
>>> weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_weight)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> for row in c6.tolist():
...     print(" ".join(f"{v:10.7f}" for v in row))
10.4130470  5.4368823  5.4368823
 5.4368823  3.0930153  3.0930153
 5.4368823  3.0930153  3.0930153
"""

from .c6 import *
from .weights import *
