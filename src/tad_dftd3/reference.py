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
Reference model
===============

This module defines the reference systems for the D3 model to compute the
C6 dispersion coefficients.
"""

import os.path as op
import torch

from .typing import Optional, Tensor


def _load_cn(dtype=torch.float):
    return torch.tensor(
        [
            [-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # None
            [+0.9118, +0.0000, -1.0000, -1.0000, -1.0000],  # H
            [+0.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # He
            [+0.0000, +0.9865, -1.0000, -1.0000, -1.0000],  # Li
            [+0.0000, +0.9808, +1.9697, -1.0000, -1.0000],  # Be
            [+0.0000, +0.9706, +1.9441, +2.9128, +4.5856],  # B
            [+0.0000, +0.9868, +1.9985, +2.9987, +3.9844],  # C
            [+0.0000, +0.9944, +2.0143, +2.9903, -1.0000],  # N
            [+0.0000, +0.9925, +1.9887, -1.0000, -1.0000],  # O
            [+0.0000, +0.9982, -1.0000, -1.0000, -1.0000],  # F
            [+0.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # Ne
            [+0.0000, +0.9684, -1.0000, -1.0000, -1.0000],  # Na
            [+0.0000, +0.9628, +1.9496, -1.0000, -1.0000],  # Mg
            [+0.0000, +0.9648, +1.9311, +2.9146, -1.0000],  # Al
            [+0.0000, +0.9507, +1.9435, +2.9407, +3.8677],  # Si
            [+0.0000, +0.9947, +2.0102, +2.9859, -1.0000],  # P
            [+0.0000, +0.9948, +1.9903, -1.0000, -1.0000],  # S
            [+0.0000, +0.9972, -1.0000, -1.0000, -1.0000],  # Cl
            [+0.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # Ar
            [+0.0000, +0.9767, -1.0000, -1.0000, -1.0000],  # K
            [+0.0000, +0.9831, +1.9349, -1.0000, -1.0000],  # Ca
            [+0.0000, +1.8627, +2.8999, -1.0000, -1.0000],  # Sc
            [+0.0000, +1.8299, +3.8675, -1.0000, -1.0000],  # Ti
            [+0.0000, +1.9138, +2.9110, -1.0000, -1.0000],  # V
            [+0.0000, +1.8269, 10.6191, -1.0000, -1.0000],  # Cr
            [+0.0000, +1.6406, +9.8849, -1.0000, -1.0000],  # Mn
            [+0.0000, +1.6483, +9.1376, -1.0000, -1.0000],  # Fe
            [+0.0000, +1.7149, +2.9263, +7.7785, -1.0000],  # Co
            [+0.0000, +1.7937, +6.5458, +6.2918, -1.0000],  # Ni
            [+0.0000, +0.9576, -1.0000, -1.0000, -1.0000],  # Cu
            [+0.0000, +1.9419, -1.0000, -1.0000, -1.0000],  # Zn
            [+0.0000, +0.9601, +1.9315, +2.9233, -1.0000],  # Ga
            [+0.0000, +0.9434, +1.9447, +2.9186, +3.8972],  # Ge
            [+0.0000, +0.9889, +1.9793, +2.9709, -1.0000],  # As
            [+0.0000, +0.9901, +1.9812, -1.0000, -1.0000],  # Se
            [+0.0000, +0.9974, -1.0000, -1.0000, -1.0000],  # Br
            [+0.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # Kr
            [+0.0000, +0.9738, -1.0000, -1.0000, -1.0000],  # Rb
            [+0.0000, +0.9801, +1.9143, -1.0000, -1.0000],  # Sr
            [+0.0000, +1.9153, +2.8903, -1.0000, -1.0000],  # Y
            [+0.0000, +1.9355, +3.9106, -1.0000, -1.0000],  # Zr
            [+0.0000, +1.9545, +2.9225, -1.0000, -1.0000],  # Nb
            [+0.0000, +1.9420, 11.0556, -1.0000, -1.0000],  # Mo
            [+0.0000, +1.6682, +9.5402, -1.0000, -1.0000],  # Tc
            [+0.0000, +1.8584, +8.8895, -1.0000, -1.0000],  # Ru
            [+0.0000, +1.9003, +2.9696, -1.0000, -1.0000],  # Rh
            [+0.0000, +1.8630, +5.7095, -1.0000, -1.0000],  # Pd
            [+0.0000, +0.9679, -1.0000, -1.0000, -1.0000],  # Ag
            [+0.0000, +1.9539, -1.0000, -1.0000, -1.0000],  # Cd
            [+0.0000, +0.9633, +1.9378, +2.9353, -1.0000],  # In
            [+0.0000, +0.9514, +1.9505, +2.9259, +3.9123],  # Sn
            [+0.0000, +0.9749, +1.9523, +2.9315, -1.0000],  # Sb
            [+0.0000, +0.9811, +1.9639, -1.0000, -1.0000],  # Te
            [+0.0000, +0.9968, -1.0000, -1.0000, -1.0000],  # I
            [+0.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # Xe
            [+0.0000, +0.9909, -1.0000, -1.0000, -1.0000],  # Cs
            [+0.0000, +0.9797, +1.8467, -1.0000, -1.0000],  # Ba
            [+0.0000, +1.9373, +2.9175, -1.0000, -1.0000],  # La
            [+2.7991, -1.0000, -1.0000, -1.0000, -1.0000],  # Ce
            [+0.0000, +2.9425, -1.0000, -1.0000, -1.0000],  # Pr
            [+0.0000, +2.9455, -1.0000, -1.0000, -1.0000],  # Nd
            [+0.0000, +2.9413, -1.0000, -1.0000, -1.0000],  # Pm
            [+0.0000, +2.9300, -1.0000, -1.0000, -1.0000],  # Sm
            [+0.0000, +1.8286, -1.0000, -1.0000, -1.0000],  # Eu
            [+0.0000, +2.8732, -1.0000, -1.0000, -1.0000],  # Gd
            [+0.0000, +2.9086, -1.0000, -1.0000, -1.0000],  # Tb
            [+0.0000, +2.8965, -1.0000, -1.0000, -1.0000],  # Dy
            [+0.0000, +2.9242, -1.0000, -1.0000, -1.0000],  # Ho
            [+0.0000, +2.9282, -1.0000, -1.0000, -1.0000],  # Er
            [+0.0000, +2.9246, -1.0000, -1.0000, -1.0000],  # Tm
            [+0.0000, +2.8482, -1.0000, -1.0000, -1.0000],  # Yb
            [+0.0000, +2.9219, -1.0000, -1.0000, -1.0000],  # Lu
            [+0.0000, +1.9254, +3.8840, -1.0000, -1.0000],  # Hf
            [+0.0000, +1.9459, +2.8988, -1.0000, -1.0000],  # Ta
            [+0.0000, +1.9292, 10.9153, -1.0000, -1.0000],  # W
            [+0.0000, +1.8104, +9.8054, -1.0000, -1.0000],  # Re
            [+0.0000, +1.8858, +9.1527, -1.0000, -1.0000],  # Os
            [+0.0000, +1.8648, +2.9424, -1.0000, -1.0000],  # Ir
            [+0.0000, +1.9188, +6.6669, -1.0000, -1.0000],  # Pt
            [+0.0000, +0.9846, -1.0000, -1.0000, -1.0000],  # Au
            [+0.0000, +1.9896, -1.0000, -1.0000, -1.0000],  # Hg
            [+0.0000, +0.9267, +1.9302, +2.9420, -1.0000],  # Tl
            [+0.0000, +0.9383, +1.9356, +2.9081, +3.9098],  # Pb
            [+0.0000, +0.9820, +1.9655, +2.9500, -1.0000],  # Bi
            [+0.0000, +0.9815, +1.9639, -1.0000, -1.0000],  # Po
            [+0.0000, +0.9954, -1.0000, -1.0000, -1.0000],  # At
            [+0.0000, -1.0000, -1.0000, -1.0000, -1.0000],  # Rn
            [+0.0000, +0.9705, -1.0000, -1.0000, -1.0000],  # Fr
            [+0.0000, +0.9662, +1.8075, -1.0000, -1.0000],  # Ra
            [+0.0000, +2.9070, -1.0000, -1.0000, -1.0000],  # Ac
            [+0.0000, +2.8844, -1.0000, -1.0000, -1.0000],  # Th
            [+0.0000, +2.8738, -1.0000, -1.0000, -1.0000],  # Pa
            [+0.0000, +2.8878, -1.0000, -1.0000, -1.0000],  # U
            [+0.0000, +2.9095, -1.0000, -1.0000, -1.0000],  # Np
            [+0.0000, +1.9209, -1.0000, -1.0000, -1.0000],  # Pu
        ]
    ).type(dtype)


def _load_c6(dtype=torch.float) -> Tensor:
    """
    Load reference C6 coefficients from file and fill them into a tensor
    """
    import math, numpy as np

    ref = torch.from_numpy(
        np.load(op.join(op.dirname(__file__), "reference-c6.npy"))
    ).type(dtype)

    n_element = (math.isqrt(8 * ref.shape[0] + 1) - 1) // 2 + 1
    n_reference = ref.shape[-1]
    c6 = torch.zeros((n_element, n_element, n_reference, n_reference), dtype=dtype)

    for i in range(1, n_element):
        for j in range(1, n_element):
            ij = i * (i - 1) // 2 + j - 1 if j < i else j * (j - 1) // 2 + i - 1
            c6[i, j, :, :] = ref[ij, :, :].T if j < i else ref[ij, :, :]

    return c6


class Reference:
    """
    Reference systems for the D3 dispersion model
    """

    c6: Tensor
    """C6 coefficients for all pairs of reference systems"""

    cn: Tensor
    """Coordination numbers for all reference systems"""

    __slots__ = [
        "c6",
        "cn",
        "__dtype",
        "__device",
    ]

    def __init__(
        self,
        cn: Optional[Tensor] = None,
        c6: Optional[Tensor] = None,
    ):

        if cn is None:
            cn = _load_cn()
        self.cn = cn
        if c6 is None:
            c6 = _load_c6()
        self.c6 = c6

        self.__dtype = self.c6.dtype
        self.__device = self.c6.device

        if any([tensor.device != self.device for tensor in (self.cn, self.c6)]):
            raise RuntimeError("All tensors must be on the same device!")

        if any([tensor.dtype != self.dtype for tensor in (self.cn, self.c6)]):
            raise RuntimeError("All tensors must have the same dtype!")

        if any(
            (
                self.c6.shape[-2] != self.c6.shape[-1],
                self.c6.shape[-1] != self.cn.shape[-1],
                self.c6.shape[-4] != self.c6.shape[-3],
                self.c6.shape[-3] != self.cn.shape[-2],
            )
        ):
            raise RuntimeError("`c6` & `cn` size mismatch found")

    @property
    def device(self) -> torch.device:
        """The device on which the `Reference` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by reference object."""
        return self.__dtype

    def to(self, device: torch.device) -> "Reference":
        """
        Returns a copy of the `Reference` instance on the specified device.

        This method creates and returns a new copy of the `Reference` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        Reference
            A copy of the `Reference` instance placed on the specified device.

        Notes
        -----
        If the `Reference` instance is already on the desired device `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(
            self.cn.to(device=device),
            self.c6.to(device=device),
        )

    def type(self, dtype: torch.dtype) -> "Reference":
        """
        Returns a copy of the `Reference` instance with specified floating point type.
        This method creates and returns a new copy of the `Reference` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Type of the

        Returns
        -------
        Reference
            A copy of the `Reference` instance with the specified dtype.

        Notes
        -----
        If the `Reference` instance has already the desired dtype `self` will be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__(
            self.cn.type(dtype),
            self.c6.type(dtype),
        )

    def __repr__(self):
        """Creates a string representation of the Reference object."""
        return f"{self.__class__.__name__}(n_element={self.cn.shape[-2]}, n_reference={self.cn.shape[-1]}, dtype={self.__dtype}, device={self.__device})"
