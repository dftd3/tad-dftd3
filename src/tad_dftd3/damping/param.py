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
==================

The parameters of a functional, as one set shared by the two- and the
three-body damping, like ``param_type`` of ``dftd``: the three-body damping
reads some of the values of the two-body damping (e.g. ``a1`` and ``a2``),
and a set from the database combines with any damping. Which values a
damping needs is declared by the damping itself, see
:mod:`tad_dftd3.damping.base`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import Tensor

__all__ = ["DampingParam", "as_damping_param"]


class DampingParam(Node):
    """
    DFT-D3 damping parameters, a set shared by the two- and three-body
    damping.

    A frozen :class:`~tad_mctc.tree.Node`: a tensor field is a pytree leaf,
    which can be differentiated or batched with ``torch.func``, and a Python
    number or ``None`` is static. A field that is ``None`` is not set: a
    damping that has a default for it uses that, one that needs it raises
    a :class:`ValueError` when it reads it.

    Parameters
    ----------
    s6, s8, s9 : Tensor | float | None, optional
        Scaling of the C6, the C8 and the three-body term. Without `s9`, or
        with ``s9=0.0`` as a Python number, the default damping has no
        three-body term.
    a1, a2 : Tensor | float | None, optional
        Scaling and offset of the critical radius of the rational damping.
        The dampings ported from dftd scale the damping radius of the model
        by them, ``a1 * rdamp + a2``, and default them to 1 and 0.
    a3, a4 : Tensor | float | None, optional
        Steepness and position of the error-function switch of the screened
        damping (as in dftd; CSO damping reads its ``a3``, ``a4`` from `rs6`
        and `rs8`, as s-dftd3).
    rs6, rs8 : Tensor | float | None, optional
        Scaling of the radii of zero damping. ``a3`` and ``a4`` of CSO damping.
    rs9 : Tensor | float | None, optional
        Scaling of the van-der-Waals radii in the three-body damping.
    alp : Tensor | float | None, optional
        Exponent of the zero damping, also of the three-body damping.
    bet : Tensor | float | None, optional
        Offset of the modified zero damping, power of the optimized power
        damping.
    sxc, rsxc : Tensor | float | None, optional
        Scaling and radius scaling of the exchange-correlation screening of
        the Koide damping. Without `sxc`, or with ``sxc=0.0`` as a Python
        number, there is no screening.
    damping : str | None, optional
        Name of the damping variant these parameters belong to, e.g.
        ``"zero"``, see :func:`tad_dftd3.damping.damping_from_name`. Static.
        If a damping is not given to :func:`tad_dftd3.dftd3`, this is the
        one that is used, and the rational damping if it is not set.
    """

    s6: Tensor | float | None = child(default=None)
    s8: Tensor | float | None = child(default=None)
    s9: Tensor | float | None = child(default=None)
    a1: Tensor | float | None = child(default=None)
    a2: Tensor | float | None = child(default=None)
    a3: Tensor | float | None = child(default=None)
    a4: Tensor | float | None = child(default=None)
    rs6: Tensor | float | None = child(default=None)
    rs8: Tensor | float | None = child(default=None)
    rs9: Tensor | float | None = child(default=None)
    alp: Tensor | float | None = child(default=None)
    bet: Tensor | float | None = child(default=None)
    sxc: Tensor | float | None = child(default=None)
    rsxc: Tensor | float | None = child(default=None)
    damping: str | None = context(default=None)

    @classmethod
    def from_functional(
        cls,
        functional: str,
        damping: str | list[str] | None = None,
        atm: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> DampingParam:
        """
        Load the parameters of a functional.

        Parameters
        ----------
        functional : str
            Name of the functional, see
            :func:`tad_dftd3.param.get_functional_params`.
        damping : str | list[str] | None, optional
            Damping variant(s) to look up, e.g. ``"bj"`` or ``"zero"``.
            Defaults to the preference of the data base, usually ``"bj"``.
        atm : bool, optional
            Whether to include the three-body term. If not, `s9` is not set.
            Defaults to ``True``.
        dtype : torch.dtype | None, optional
            Floating point precision of the tensors.
        device : torch.device | None, optional
            Device of the tensors.

        Returns
        -------
        DampingParam
            The parameters, as tensors.

        Raises
        ------
        KeyError
            If the functional is unknown.
        ValueError
            If there are no parameters of that damping for it.
        """
        from ..param import get_functional_params

        values: dict[str, Any] = {
            **get_functional_params(
                functional,
                damping=damping,
                keep_meta=True,
                dtype=dtype,
                device=device,
            )
        }
        for key in ("mbd", "doi"):
            values.pop(key, None)
        if not atm:
            del values["s9"]
        return cls(**values)


def as_damping_param(param: DampingParam | Mapping[str, Any]) -> DampingParam:
    """
    The parameters as a :class:`DampingParam`: one is returned as it is, a
    dictionary is unpacked into one, so an unknown name is a
    :class:`TypeError`. A name it leaves out is not set; nothing is filled
    in.
    """
    if isinstance(param, DampingParam):
        return param
    return DampingParam(**param)
