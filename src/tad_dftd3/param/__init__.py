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
Damping parameters
===================

Functional-specific D3 damping parameters, sourced from s-dftd3's
``parameters.toml`` data base (vendored verbatim as ``parameters.toml`` in
this subpackage). Functional-name aliasing, which upstream hardcodes as a
Fortran ``select case`` in ``get_method_id``, is derived programmatically
into ``aliases.toml`` by ``tools/gen_param_aliases.py`` and applied here.
"""

from .functional import *
