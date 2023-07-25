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
Test the utility functions.
"""

import torch

from tad_dftd3 import util


def test_real_atoms() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0, 0, 0],  # H2
            [6, 1, 1, 1, 1],  # CH4
        ],
    )
    ref = torch.tensor(
        [
            [True, True, False, False, False],  # H2
            [True, True, True, True, True],  # CH4
        ],
    )
    mask = util.real_atoms(numbers)
    assert (mask == ref).all()


def test_real_pairs_single() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1])  # CH4
    size = numbers.shape[0]

    ref = torch.full((size, size), True)
    mask = util.real_pairs(numbers, diagonal=True)
    assert (mask == ref).all()

    ref *= ~torch.diag_embed(torch.ones(size, dtype=torch.bool))
    mask = util.real_pairs(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_pairs_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
            ],
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
        ]
    )
    mask = util.real_pairs(numbers, diagonal=True)
    assert (mask == ref).all()

    ref = torch.tensor(
        [
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ],
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
        ]
    )
    mask = util.real_pairs(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_triples_single() -> None:
    numbers = torch.tensor([8, 1, 1])  # H2O
    size = numbers.shape[0]

    ref = torch.full((size, size, size), True)
    mask = util.real_triples(numbers, diagonal=True)
    assert (mask == ref).all()

    ref *= ~torch.diag_embed(torch.ones(size, dtype=torch.bool))
    mask = util.real_pairs(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_triples_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ],
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
            ],
        ]
    )
    mask = util.real_triples(numbers, diagonal=True)
    assert (mask == ref).all()

    ref = torch.tensor(
        [
            [
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
            ],
        ]
    )
    mask = util.real_triples(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_triples_self_single() -> None:
    numbers = torch.tensor([8, 1, 1])  # H2O

    ref = torch.tensor(
        [
            [
                [False, False, False],
                [False, False, True],
                [False, True, False],
            ],
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ],
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ],
        ],
        dtype=torch.bool,
    )

    mask = util.real_triples(numbers, self=False)
    assert (mask == ref).all()


def test_real_triples_self_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                ],
                [
                    [False, False, True],
                    [False, False, False],
                    [True, False, False],
                ],
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
            ],
        ]
    )

    mask = util.real_triples(numbers, self=False)
    assert (mask == ref).all()


def test_pack() -> None:
    mol1 = torch.tensor([1, 1])  # H2
    mol2 = torch.tensor([8, 1, 1])  # H2O

    # dummy test: only give single tensor
    assert (mol1 == util.pack(mol1)).all()

    # standard packing
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )
    packed = util.pack([mol1, mol2])
    assert (packed == ref).all()

    # different axis
    packed = util.pack([mol1, mol2], axis=-1)
    assert (packed == ref.T).all()
