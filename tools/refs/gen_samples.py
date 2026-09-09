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
Regenerate the ``refs`` dict in ``test/test_model/samples.py``. See
README.md in this directory for when and how to run this.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tad_mctc.data.molecules import mols
from tad_mctc.ncoord.defaults import CUTOFF_D3

TOOL = Path(__file__).resolve().parent / "dump_reference"
MAX_REFERENCE_SLOTS = 7
SAMPLE_LIST = ["SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


def run_tool(
    numbers: list[int], positions: list[list[float]], cn_cutoff: float
):
    lines = [str(len(numbers)), repr(cn_cutoff)]
    for z, (x, y, zz) in zip(numbers, positions):
        lines.append(f"{z} {x!r} {y!r} {zz!r}")
    result = subprocess.run(
        [str(TOOL)],
        input="\n".join(lines),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def fmt(x: float) -> str:
    """
    Format one float the way this repository's other ``samples.py`` files
    do: a signed mantissa with 14 decimal digits, and a signed exponent only
    if negative (``+1.51435152661277e02``, ``-1.01377874949909e-06``).
    """
    mantissa, exponent = f"{x:+.14e}".split("e")
    sign, digits = exponent[0], exponent[1:]
    return f"{mantissa}e{digits}" if sign == "+" else f"{mantissa}e-{digits}"


def fmt_vector(values: list[float], indent: str) -> str:
    body = ",\n".join(f"{indent}    {fmt(v)}" for v in values)
    return "[\n" + body + f",\n{indent}]"


def fmt_matrix(rows: list[list[float]], indent: str) -> str:
    row_indent = indent + "    "
    body = "\n".join(
        f"{row_indent}{fmt_vector(row, row_indent)}," for row in rows
    )
    return "[\n" + body + f"\n{indent}]"


def main() -> None:
    print("refs: Dict[str, Refs] = {")
    for name in SAMPLE_LIST:
        mol = mols[name]
        numbers = mol["numbers"].tolist()
        positions = mol["positions"].tolist()
        data = run_tool(numbers, positions, CUTOFF_D3)

        mref = len(data["weights"][0])
        weights = [
            row + [0.0] * (MAX_REFERENCE_SLOTS - mref)
            for row in data["weights"]
        ]

        print(f'    "{name}": Refs(')
        print("        {")
        print(f'            "cn": torch.tensor(')
        print(f"                {fmt_vector(data['cn'], ' ' * 16)},")
        print("                dtype=torch.double,")
        print("            ),")
        print(f'            "weights": torch.tensor(')
        print(f"                {fmt_matrix(weights, ' ' * 16)},")
        print("                dtype=torch.double,")
        print("            ),")
        print(f'            "c6": torch.tensor(')
        print(f"                {fmt_matrix(data['c6'], ' ' * 16)},")
        print("                dtype=torch.double,")
        print("            ),")
        print("        }")
        print("    ),")
    print("}")


if __name__ == "__main__":
    main()
