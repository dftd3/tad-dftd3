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
Coordination Number
===================

Functions for calculating the D3 coordination numbers.
Only exported for convenience.

.. note::

    :func:`tad_mctc.ncoord.d3.cn_d3` applies tad-mctc's generic cutoff
    (:data:`tad_mctc.ncoord.defaults.CUTOFF_D3`), not the D3-specific one
    :func:`tad_dftd3.disp.dftd3` uses
    (:data:`tad_dftd3.defaults.D3_CN_CUTOFF`, s-dftd3's own). Both are
    faithful to their respective Fortran source. To reproduce
    :func:`~tad_dftd3.disp.dftd3`, pass ``cutoff`` to
    :func:`tad_mctc.ncoord.coordination_number` explicitly.
"""

from tad_mctc.ncoord import coordination_number
from tad_mctc.ncoord.count import exp_count
from tad_mctc.ncoord.d3 import cn_d3

__all__ = ["cn_d3", "coordination_number", "exp_count"]
