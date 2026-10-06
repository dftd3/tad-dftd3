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
Test the reference.
"""

from typing import Any, TypedDict
from unittest.mock import patch

import pytest
import torch
from tad_mctc import Structure
from tad_mctc.convert import str_to_device
from tad_mctc.typing import DD, MockTensor, Tensor

from tad_dftd3 import reference

from ..conftest import DEVICE

sample_list = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_dtype(dtype: torch.dtype) -> None:
    ref = reference.Reference.load().type(dtype)
    assert ref.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, None])
def test_reference_dtype_both(dtype: torch.dtype | None) -> None:
    class DDNone(TypedDict):
        device: torch.device
        dtype: torch.dtype | None

    dev = torch.device("cpu")
    dd: DDNone = {"device": dev, "dtype": dtype}
    ref = reference.Reference.load(device=dev).to(**dd)
    assert ref.dtype == torch.tensor(1.0, dtype=dtype).dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_move_both(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    ref = reference.Reference.load(device=DEVICE).to(**dd)
    assert ref.dtype == dtype


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
@pytest.mark.parametrize("device_str2", ["cpu", "cuda"])
def test_reference_device(device_str: str, device_str2: str) -> None:
    device = str_to_device(device_str)
    device2 = str_to_device(device_str2)
    ref = reference.Reference.load(device=device2).to(device)
    assert ref.device == device

    with pytest.raises(AttributeError):
        ref.device = device  # type: ignore[misc]


def test_reference_different_devices() -> None:
    cn = reference._load_cn()  # pylint: disable=protected-access
    c6 = reference._load_c6().to("meta")  # pylint: disable=protected-access
    with pytest.raises(RuntimeError, match="different devices"):
        reference.Reference(cn=cn, c6=c6)


def test_reference_fail() -> None:
    cn = reference._load_cn()  # pylint: disable=protected-access
    c6 = reference._load_c6()  # pylint: disable=protected-access

    # wrong dtype
    with pytest.raises(TypeError, match="different dtypes"):
        reference.Reference(cn=cn, c6=c6.type(torch.float16))

    # wrong shape
    with pytest.raises(ValueError, match="size mismatch"):
        reference.Reference(
            cn=torch.rand((4, 4), dtype=c6.dtype, device=c6.device), c6=c6
        )

    ref = reference.Reference.load(
        device=torch.device("cpu"), dtype=torch.float64
    )
    assert (
        repr(ref)
        == "Reference(n_element=104, n_reference=7, dtype=torch.float64, device=cpu)"
    )


def test_default_reference_shares_memory() -> None:
    """
    The read-only default reference of `dftd3` does not copy the C6
    coefficients, while a public `Reference` does.
    """
    # pylint: disable=protected-access
    dd: DD = {"device": torch.device("cpu"), "dtype": torch.float64}

    shared = reference._default_reference(torch.empty(0, **dd))
    copied = reference.Reference.load(**dd)

    assert shared.c6.data_ptr() == reference._C6.data_ptr()
    assert copied.c6.data_ptr() != reference._C6.data_ptr()

    assert torch.equal(shared.c6, copied.c6)
    assert torch.equal(shared.cn, copied.cn)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_reference_dtype(dtype: torch.dtype) -> None:
    """
    On any dtype the default reference matches a `Reference`, and it is
    built once per device and dtype, not per call.
    """
    # pylint: disable=protected-access
    dd: DD = {"device": DEVICE, "dtype": dtype}

    shared = reference._default_reference(torch.empty(0, **dd))
    copied = reference.Reference.load(**dd)

    assert shared.dtype == dtype
    assert torch.equal(shared.c6, copied.c6)
    assert torch.equal(shared.cn, copied.cn)

    again = reference._default_reference(torch.empty(0, **dd))
    assert again.c6.data_ptr() == shared.c6.data_ptr()
    assert again.cn.data_ptr() == shared.cn.data_ptr()


def test_dftd3_does_not_copy_reference() -> None:
    """`dftd3` builds its default reference without copying the C6 table."""
    # pylint: disable=protected-access
    from tad_dftd3 import dftd3

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    param = {
        "s8": torch.tensor(1.0),
        "a1": torch.tensor(0.4),
        "a2": torch.tensor(4.6),
    }

    structure = Structure(numbers=numbers, positions=positions)
    with patch(
        "tad_dftd3.reference._load_c6", side_effect=AssertionError("copied")
    ):
        energy = dftd3(structure, param)

    ref = dftd3(
        structure,
        param,
        ref=reference.Reference.load(dtype=torch.float64),
    )
    assert pytest.approx(ref.cpu()) == energy.cpu()
