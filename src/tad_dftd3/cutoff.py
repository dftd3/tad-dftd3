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
Real-space cutoffs
==================

One real-space cutoff per part of the D3 model, since they decay at very
different rates: the counting function behind the coordination number
exponentially, the two-body energy as :math:`R^{-6}`, the three-body term
as :math:`R^{-9}`.

Mirrors ``realspace_cutoff`` in s-dftd3's ``dftd3_cutoff``, defaults
included, so both implementations discard the same pairs and triples.
s-dftd3 can also switch the dispersion terms off smoothly over a width
(``width2``/``width3``); those default to zero there, i.e. the hard cutoff
implemented here.

Example
-------
>>> from tad_dftd3.cutoff import Cutoff
>>> cutoff = Cutoff()
>>> print(cutoff)
Cutoff(cn=40.0, disp2=60.0, disp3=40.0)
>>> cutoff = Cutoff(disp3=25.0)
>>> print(float(cutoff.disp3))
25.0
"""

from __future__ import annotations

import torch
from tad_mctc.typing import DD, Tensor, TensorLike

from . import defaults

__all__ = ["Cutoff"]


class Cutoff(TensorLike):
    """
    Real-space cutoffs for the individual parts of the D3 model, in Bohr.
    """

    cn: Tensor
    """Coordination number cutoff."""

    disp2: Tensor
    """Two-body dispersion interaction cutoff."""

    disp3: Tensor
    """Three-body dispersion interaction cutoff."""

    __slots__ = ["cn", "disp2", "disp3"]

    def __init__(
        self,
        cn: int | float | Tensor = defaults.D3_CN_CUTOFF,
        disp2: int | float | Tensor = defaults.D3_DISP2_CUTOFF,
        disp3: int | float | Tensor = defaults.D3_DISP3_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiate the collection of real-space cutoffs.

        Parameters
        ----------
        cn : int | float | Tensor, optional
            Coordination number cutoff. Defaults to
            :data:`tad_dftd3.defaults.D3_CN_CUTOFF`.
        disp2 : int | float | Tensor, optional
            Two-body dispersion interaction cutoff. Defaults to
            :data:`tad_dftd3.defaults.D3_DISP2_CUTOFF`.
        disp3 : int | float | Tensor, optional
            Three-body dispersion interaction cutoff. Defaults to
            :data:`tad_dftd3.defaults.D3_DISP3_CUTOFF`.
        device : :class:`torch.device` | None, optional
            Device to store the tensors on. Defaults to ``None``.
        dtype : :class:`torch.dtype` | None, optional
            Floating point dtype of the tensors. Defaults to ``None``.
        """
        super().__init__(device, dtype)
        dd: DD = {"device": self.device, "dtype": self.dtype}

        self.cn = self._as_tensor(cn, "cn", dd)
        self.disp2 = self._as_tensor(disp2, "disp2", dd)
        self.disp3 = self._as_tensor(disp3, "disp3", dd)

    def _as_tensor(
        self, value: int | float | Tensor, name: str, dd: DD
    ) -> Tensor:
        """Cast one cutoff to a scalar tensor; ``name`` labels the error."""
        if isinstance(value, Tensor):
            tensor = value.to(**dd)
        else:
            tensor = torch.tensor(value, **dd)

        if tensor.ndim != 0:
            raise ValueError(
                f"Cutoff '{name}' must be a scalar, but has shape "
                f"{tuple(tensor.shape)}."
            )

        return tensor

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(cn={float(self.cn)}, "
            f"disp2={float(self.disp2)}, disp3={float(self.disp3)})"
        )

    def __repr__(self) -> str:
        return str(self)
