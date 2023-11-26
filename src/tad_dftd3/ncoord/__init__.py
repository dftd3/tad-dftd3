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
Coordination number
===================

Evaluates a fractional coordination number for a given geometry or batch of geometries.

Examples
--------
>>> import torch
>>> import tad_dftd3 as d3
>>> numbers = d3.util.pack((
...     torch.tensor([7, 1, 1, 1]),
...     torch.tensor([6, 8, 7, 1, 1, 1]),
...     torch.tensor([6, 8, 8, 1, 1]),
... ))
>>> positions = d3.util.pack((
...     torch.tensor([
...         [+0.00000000000000, +0.00000000000000, -0.54524837997150],
...         [-0.88451840382282, +1.53203081565085, +0.18174945999050],
...         [-0.88451840382282, -1.53203081565085, +0.18174945999050],
...         [+1.76903680764564, +0.00000000000000, +0.18174945999050],
...     ]),
...     torch.tensor([
...         [-0.55569743203406, +1.09030425468557, +0.00000000000000],
...         [+0.51473634678469, +3.15152550263611, +0.00000000000000],
...         [+0.59869690244446, -1.16861263789477, +0.00000000000000],
...         [-0.45355203669134, -2.74568780438064, +0.00000000000000],
...         [+2.52721209544999, -1.29200800956867, +0.00000000000000],
...         [-2.63139587595376, +0.96447869452240, +0.00000000000000],
...     ]),
...     torch.tensor([
...         [-0.53424386915034, -0.55717948166537, +0.00000000000000],
...         [+0.21336223456096, +1.81136801357186, +0.00000000000000],
...         [+0.82345103924195, -2.42214694643037, +0.00000000000000],
...         [-2.59516465056138, -0.70672678063558, +0.00000000000000],
...         [+2.09259524590881, +1.87468519515944, +0.00000000000000],
...     ]),
... ))
>>> rcov = d3.data.covalent_rad_d3[numbers]
>>> cn = d3.ncoord.coordination_number(numbers, positions, rcov, d3.ncoord.exp_count)
>>> torch.set_printoptions(precision=7)
>>> print(cn)
tensor([[2.9901006, 0.9977214, 0.9977214, 0.9977214, 0.0000000, 0.0000000],
        [3.0059586, 1.0318390, 3.0268824, 1.0061584, 1.0036336, 0.9989871],
        [3.0093639, 2.0046251, 1.0187057, 0.9978270, 1.0069743, 0.0000000]])
"""
from .count import dexp_count, exp_count
from .d3 import coordination_number_d3 as coordination_number
