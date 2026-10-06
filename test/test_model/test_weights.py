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
Test the weights.
"""

import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.typing import DD

from tad_dftd3 import model, reference

from ..conftest import DEVICE
from ..references import reference_cn, reference_weights
from ..utils import approx_ref, load_sample

sample_list: list[tuple[str, str]] = [
    ("mb16_43", "SiH4"),
    ("heavy28", "pbh4_bih3"),
    ("other", "C6H5I-CH3SH"),
    ("mb16_43", "01"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source", sample_list)
def test_single(dtype: torch.dtype, source: tuple[str, str]) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = load_sample(*source, dd)[0]
    ref = reference.Reference.load(**dd)

    # coordination number and weights both come from s-dftd3's Fortran
    # library (see test/references)
    cn = reference_cn(*source, dd)
    refgw = reference_weights(*source, dd)

    weights = model.weight_references(
        numbers, cn, ref, model.gaussian_log_weight
    )

    assert weights.dtype == dtype
    assert approx_ref(refgw.cpu(), dtype) == weights.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("source1", [("mb16_43", "SiH4")])
@pytest.mark.parametrize("source2", sample_list)
def test_batch(
    dtype: torch.dtype, source1: tuple[str, str], source2: tuple[str, str]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = pack(
        (
            load_sample(*source1, dd)[0],
            load_sample(*source2, dd)[0],
        )
    )
    ref = reference.Reference.load(**dd)
    # s-dftd3 has no notion of a batch of independent molecules; the
    # reference of each molecule is packed like the inputs above
    cn = pack((reference_cn(*source1, dd), reference_cn(*source2, dd)))
    refgw = pack(
        (reference_weights(*source1, dd), reference_weights(*source2, dd))
    )

    weights = model.weight_references(
        numbers, cn, ref, model.gaussian_log_weight
    )

    assert weights.dtype == dtype
    assert approx_ref(refgw.cpu(), dtype) == weights.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    ("number", "cn"),
    [(11, 14.4), (2, 30.0)],
    ids=["subnormal", "zero"],
)
def test_cn_far_from_all_references(
    dtype: torch.dtype, number: int, cn: float
) -> None:
    """
    An atom whose CN is far above its references, so that its Gaussian
    weights underflow (in float64 to subnormal numbers for Na, references
    at CN 0 and 0.97, and to zero for He, one reference at CN 0): the
    closest reference takes all the weight, and the gradient is finite and
    as small as the weights of the others, not NaN.
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    ref = reference.Reference.load(**dd)
    numbers = torch.tensor([number, 6], device=DEVICE)
    cns = torch.tensor([cn, 3.0], **dd, requires_grad=True)

    weights = model.weight_references(numbers, cns, ref)
    # the reference CNs, weighted: depends on the weights of each reference
    refcn = torch.where(weights > 0, ref.cn[numbers], 0.0)
    (grad,) = torch.autograd.grad((weights * refcn).sum(), cns)

    closest = int(torch.argmax(ref.cn[number]))
    assert weights[0, closest] == 1.0
    assert torch.isfinite(grad).all()
    assert abs(grad[0].item()) < 1e-30


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_padding_atom_has_no_weights(dtype: torch.dtype) -> None:
    """A padding atom (number 0) has no reference, so all its weights are
    zero, and so is its gradient."""
    dd: DD = {"device": DEVICE, "dtype": dtype}
    ref = reference.Reference.load(**dd)
    numbers = torch.tensor([6, 0], device=DEVICE)
    cns = torch.tensor([3.0, 0.0], **dd, requires_grad=True)

    weights = model.weight_references(numbers, cns, ref)
    (grad,) = torch.autograd.grad(weights.sum(), cns)

    assert (weights[1] == 0).all()
    assert torch.isfinite(grad).all()
