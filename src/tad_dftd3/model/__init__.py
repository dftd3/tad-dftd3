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
>>> positions = torch.Tensor([
...     [+0.00000000000000, +0.00000000000000, -0.73578586109551],
...     [+1.44183152868459, +0.00000000000000, +0.36789293054775],
...     [-1.44183152868459, +0.00000000000000, +0.36789293054775],
... ])
>>> ref = d3.reference.Reference()
>>> rcov = d3.data.covalent_rad_d3[numbers]
>>> cn = mctc.ncoord.cn_d3(numbers, positions, rcov=rcov, counting_function=d3.ncoord.exp_count)
>>> weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_weight)
>>> c6 = d3.model.atomic_c6(numbers, weights, ref)
>>> torch.set_printoptions(precision=7)
>>> print(c6)
tensor([[10.4130471,  5.4368822,  5.4368822],
        [ 5.4368822,  3.0930154,  3.0930154],
        [ 5.4368822,  3.0930154,  3.0930154]], dtype=torch.float64)
"""
from .c6 import *
from .weights import *
