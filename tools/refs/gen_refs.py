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
Regenerate ``test/references/``, one JSON file per molecule. See README.md
in this directory for when and how to run this.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tad_mctc.data.molecules import mols

from tad_dftd3.defaults import D3_CN_CUTOFF

HERE = Path(__file__).resolve().parent
# fpm's and Meson's own, already-fixed install paths -- see README.md for
# how to build either one; whichever exists is used.
TOOL_CANDIDATES = [
    HERE / "_install_fpm" / "bin" / "gen_refs_fortran",
    HERE / "_install_meson" / "bin" / "gen_refs_fortran",
]
OUT_DIR = Path(__file__).resolve().parents[2] / "test" / "references"

MAX_REFERENCE_SLOTS = 7

SAMPLE_LIST = ["AmF3", "SiH4", "PbH4-BiH3", "C6H5I-CH3SH", "MB16_43_01"]


def find_tool() -> Path:
    for candidate in TOOL_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "gen_refs_fortran is not built -- see README.md in this directory "
        f"(looked in {', '.join(str(c) for c in TOOL_CANDIDATES)})"
    )


def run_tool(
    tool: Path, numbers: list[int], positions: list[list[float]], cn_cutoff: float
):
    lines = [str(len(numbers)), repr(cn_cutoff)]
    for z, (x, y, zz) in zip(numbers, positions):
        lines.append(f"{z} {x!r} {y!r} {zz!r}")
    result = subprocess.run(
        [str(tool)],
        input="\n".join(lines),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    tool = find_tool()
    for name in SAMPLE_LIST:
        mol = mols[name]
        numbers = mol["numbers"].tolist()
        positions = mol["positions"].tolist()
        data = run_tool(tool, numbers, positions, D3_CN_CUTOFF)

        mref = len(data["weights"][0])
        weights = [
            row + [0.0] * (MAX_REFERENCE_SLOTS - mref)
            for row in data["weights"]
        ]

        model_data = {
            "cn": data["cn"],
            "weights": weights,
            "c6": data["c6"],
        }

        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(model_data, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
