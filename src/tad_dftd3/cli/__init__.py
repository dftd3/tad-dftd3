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
Command line interface
======================

`tad_dftd3` as a command line tool: read a structure file and print its
DFT-D3 dispersion energy, split into the two- and three-body term, and
optionally its gradient and the wall time of every step of the pipeline
(neighbour search, coordination number, reference weights, C6
coefficients, each energy term, backward pass).

.. code-block:: sh

   tad_dftd3 --func pbe0 structure.xyz
   tad_dftd3 --func b3lyp --damping zero --no-atm --grad --timing coord
   tad_dftd3 --func pbe0 --neighbor dense --cuda --omp 4 POSCAR
   tad_dftd3 --func pbe0 --nlist-only --timing structure.xyz

The package splits into the argument parser (:mod:`._args`), the printed
sections (:mod:`._output`) and the run itself (:mod:`._main`). The step
timer and the sections shared with it are those of tad-mctc's command line
(:mod:`tad_mctc.cli`).
"""

from ._main import main

__all__ = ["main"]
