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
Test the command line tool (`tad_dftd3.cli`): the energy and gradient it
prints are those of :func:`tad_dftd3.dftd3` on the structure read from the
same file, every step of the pipeline is timed on its own, and bad
arguments are usage errors.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
import torch
from tad_mctc.data import pse
from tad_mctc.data.structures import get_structure
from tad_mctc.io.read import read_structure
from tad_mctc.io.structure import Structure
from tad_mctc.io.write import write_xyz
from tad_mctc.tools.testing import requires_compile
from tad_mctc.typing import Tensor

import tad_dftd3 as d3
from tad_dftd3.cli import main

_AA_PER_BOHR = 0.529177210903

_WATER = """3
water
O  0.000  0.000  0.100
H  0.757  0.586  0.000
H -0.757  0.586  0.000
"""

_WATER_AND_HYDROGEN = """3
water
O  0.000  0.000  0.100
H  0.757  0.586  0.000
H -0.757  0.586  0.000
2
hydrogen
H  0.000  0.000  0.100
H  0.000  0.000  0.840
"""


def _molecule(tmp_path: Path) -> Path:
    """A molecule of the MB16-43 set, written as xyz."""
    structure = get_structure("mb16_43", "01", dtype=torch.double)
    path = tmp_path / "mol.xyz"
    write_xyz(path, structure.numbers, structure.positions)
    return path


_CELL_DISP3 = 15.0
"""Three-body cutoff of the runs on a cell: at the default, the dense
three-body term of a cell takes minutes."""


def _cell(tmp_path: Path) -> Path:
    """A triclinic cell, written as a POSCAR in Cartesian coordinates."""
    structure = get_structure("other", "periodic_triclinic", dtype=torch.double)
    assert structure.lattice is not None
    symbols = [pse.Z2S[int(z)] for z in structure.numbers]
    lines = ["triclinic", "1.0"]
    lines += [
        " ".join(f"{x * _AA_PER_BOHR:.10f}" for x in row)
        for row in structure.lattice.tolist()
    ]
    lines += [" ".join(symbols), " ".join("1" for _ in symbols), "Cartesian"]
    lines += [
        " ".join(f"{x * _AA_PER_BOHR:.10f}" for x in row)
        for row in structure.positions.tolist()
    ]
    path = tmp_path / "POSCAR"
    path.write_text("\n".join(lines) + "\n")
    return path


def _values(out: str, label: str) -> list[float]:
    """The numbers printed after `label` in the results of a run."""
    results = out.split("\nResults\n")[-1].split("\nTiming\n")[0]
    (line,) = (
        line for line in results.splitlines() if line.strip().startswith(label)
    )
    return [float(v) for v in line.split()[len(label.split()) : -1]]


def _reference(
    path: Path, atm: bool, sparse: bool, cutoff: d3.Cutoff = d3.Cutoff()
) -> tuple[Tensor, Tensor]:
    """Energy per frame and gradient of `dftd3` on the structure in
    `path`, with the PBE0 parameters of the run."""
    structure = read_structure(path, dtype=torch.double)
    positions = structure.positions.detach().requires_grad_(True)
    structure = structure.replace(positions=positions)
    param = d3.DampingParam.from_functional("pbe0", atm=atm, dtype=torch.double)

    energy = d3.dftd3(structure, param, sparse=sparse, cutoff=cutoff).sum(-1)
    (gradient,) = torch.autograd.grad(energy.sum(), positions)
    return torch.atleast_1d(energy.detach()), gradient


@pytest.mark.parametrize("neighbor", ["dense", "sparse"])
@pytest.mark.parametrize("atm", [True, False])
@pytest.mark.parametrize("system", ["molecule", "cell"])
def test_energy_and_gradient_match_the_library(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    neighbor: str,
    atm: bool,
    system: str,
) -> None:
    """The printed total energy and gradient norm are those of `dftd3`."""
    cutoff = d3.Cutoff()
    if system == "molecule":
        path = _molecule(tmp_path)
    else:
        path, cutoff = _cell(tmp_path), d3.Cutoff(disp3=_CELL_DISP3)
    argv = ["--func", "pbe0", "--grad", "--neighbor", neighbor, str(path)]
    argv += ["--cutoff-disp3", str(cutoff.disp3)]
    if not atm:
        argv.insert(0, "--no-atm")

    assert main(argv) == 0
    out = capsys.readouterr().out

    sparse = neighbor == "sparse"
    energy, gradient = _reference(path, atm, sparse, cutoff)
    assert _values(out, "total") == pytest.approx(energy.tolist(), abs=1e-11)
    assert ("three-body" in out.split("Results")[-1]) == atm

    norm = float(torch.linalg.vector_norm(gradient))
    assert _values(out, "norm") == pytest.approx([norm], abs=1e-11)


def test_two_and_three_body_add_up_to_the_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The split of the energy into its terms adds up to the total."""
    assert main(["--func", "pbe0", str(_molecule(tmp_path))]) == 0
    out = capsys.readouterr().out

    (e2,), (e3,) = _values(out, "two-body"), _values(out, "three-body")
    assert e2 + e3 == pytest.approx(_values(out, "total")[0], abs=1e-11)
    assert e3 != 0.0


def test_recompute_mode_gives_the_same_gradient(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checkpointing changes the memory of the backward pass only."""
    path = str(_cell(tmp_path))
    argv = ["--func", "pbe0", "--grad", "--cutoff-disp3", str(_CELL_DISP3)]
    argv += [path]

    assert main(argv) == 0
    graph = capsys.readouterr().out
    assert main([*argv, "--mode", "recompute", "--max-triples", "5000"]) == 0
    recompute = capsys.readouterr().out

    assert _values(recompute, "total") == pytest.approx(
        _values(graph, "total"), abs=1e-12
    )
    assert _values(recompute, "norm") == pytest.approx(
        _values(graph, "norm"), abs=1e-12
    )


def test_multi_frame_file_reports_every_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A multi-frame file is read as a padded batch, with one energy per
    frame, and the gradient summary names only real atoms."""
    path = tmp_path / "frames.xyz"
    path.write_text(_WATER_AND_HYDROGEN)

    assert main(["--func", "pbe0", "--grad", str(path)]) == 0
    out = capsys.readouterr().out

    energy, _ = _reference(path, atm=True, sparse=True)
    assert "frames    2" in out
    assert _values(out, "total") == pytest.approx(energy.tolist(), abs=1e-11)
    assert len(_values(out, "norm")) == 2
    assert "frame" in out.split("max atom")[-1]


def test_cutoffs_and_widths_are_passed_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cutoffs on the command line are those of the model and of the
    neighbour lists."""
    path = _cell(tmp_path)
    argv = ["--cutoff-cn", "20", "--cutoff-disp2", "30", "--cutoff-disp3"]
    argv += ["15", "--width2", "4", "--width3", "2"]

    assert main(["--func", "pbe0", *argv, str(path)]) == 0
    out = capsys.readouterr().out

    cutoff = d3.Cutoff(cn=20.0, disp2=30.0, disp3=15.0, width2=4.0, width3=2.0)
    structure = read_structure(path, dtype=torch.double)
    param = d3.DampingParam.from_functional("pbe0", dtype=torch.double)
    energy = d3.dftd3(structure, param, cutoff=cutoff, sparse=True).sum()

    assert _values(out, "total") == pytest.approx([float(energy)], abs=1e-11)
    lists = out.split("Neighbour lists")[-1]
    for term, value in (("cn", "20.0000"), ("disp2", "30.0000")):
        assert f"{term:<6}  {value:>8}" in lists


def test_damping_variant_is_chosen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--damping zero` loads the zero-damping parameters."""
    path = _molecule(tmp_path)

    assert main(["--func", "pbe0", "--damping", "zero", str(path)]) == 0
    out = capsys.readouterr().out

    structure = read_structure(path, dtype=torch.double)
    param = d3.DampingParam.from_functional(
        "pbe0", damping="zero", dtype=torch.double
    )
    energy = d3.dftd3(structure, param).sum()

    assert "ZeroTwoBody" in out
    assert _values(out, "total") == pytest.approx([float(energy)], abs=1e-11)


@pytest.mark.parametrize("neighbor", ["dense", "sparse"])
@pytest.mark.parametrize("system", ["molecule", "cell"])
def test_every_step_is_timed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    neighbor: str,
    system: str,
) -> None:
    """The coordination number, the weights, the C6 coefficients, each
    term and the backward pass are separate rows of the timing table, as
    is the build of the neighbour lists or of a cell's image shifts."""
    path = _molecule(tmp_path) if system == "molecule" else _cell(tmp_path)
    argv = ["--timing", "--grad", "--neighbor", neighbor, "--func", "pbe0"]
    argv += ["--cutoff-disp3", str(_CELL_DISP3)]

    assert main([*argv, str(path)]) == 0
    table = capsys.readouterr().out.split("Timing")[-1]

    steps = [
        "read structure",
        "load reference",
        f"coordination number ({neighbor})",
        "reference weights",
        "atomic C6",
        f"two-body ({neighbor})",
        f"three-body ({neighbor})",
        "gradient (backward)",
    ]
    if neighbor == "sparse":
        steps += ["native extension", "nlist: pair filter"]
    else:
        steps += ["C6 matrix"]
    if neighbor == "dense" and system == "cell":
        steps += [f"periodic shifts ({t})" for t in ("cn", "disp2", "disp3")]

    for step in steps:
        assert step in table
    assert ("C6 matrix" in table) == (neighbor == "dense")
    assert ("nlist:" in table) == (neighbor == "sparse")


@pytest.mark.parametrize("atm", [True, False])
def test_nlist_only_builds_one_list_per_term(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], atm: bool
) -> None:
    """Without the three-body term, there is no list for it."""
    argv = ["--func", "pbe0", "--nlist-only", str(_cell(tmp_path))]
    if not atm:
        argv.insert(0, "--no-atm")

    assert main(argv) == 0
    out = capsys.readouterr().out

    lists = out.split("Neighbour lists")[-1]
    assert "cn " in lists and "disp2 " in lists
    assert ("disp3 " in lists) == atm
    assert "Results" not in out


def test_dense_run_does_not_load_the_native_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dense path builds no neighbour list, so it neither loads the
    extension nor reports it, nor prints lists."""
    path = tmp_path / "water.xyz"
    path.write_text(_WATER)

    argv = ["--timing", "--neighbor", "dense", "--func", "pbe0", str(path)]
    assert main(argv) == 0
    out = capsys.readouterr().out

    assert "native extension" not in out
    assert "Native extension" not in out
    assert "Neighbour lists" not in out


def test_coldfusion_check_is_a_timed_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "water.xyz"
    path.write_text(_WATER)

    argv = ["--timing", "--coldfusion-check", "--func", "pbe0", str(path)]
    assert main(argv) == 0

    assert "cold-fusion check" in capsys.readouterr().out.split("Timing")[-1]


def test_omp_setting_is_restored_after_the_run(tmp_path: Path) -> None:
    """`--omp` applies to the run only."""
    path = tmp_path / "water.xyz"
    path.write_text(_WATER)
    before = torch.get_num_threads()

    argv = ["--omp", str(before + 1), "--func", "pbe0", str(path)]
    assert main(argv) == 0

    assert torch.get_num_threads() == before


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--func", "nope"], "unknown functional 'nope'"),
        (
            ["--func", "slaterdirac", "--damping", "bj"],
            "none of the requested",
        ),
        (["--func", "pbe0", "--cutoff-cn", "-1"], "must not be negative"),
        (["--func", "pbe0", "--width2", "x"], "invalid float value: 'x'"),
        (["--func", "pbe0", "--max-triples", "0"], "must be at least 1"),
        (["--func", "pbe0", "--nlist-only", "--grad"], "no energy"),
        (
            ["--func", "pbe0", "--compile", "--mode", "recompute"],
            "cannot be combined with --compile",
        ),
        (
            ["--func", "pbe0", "--nlist-only", "--neighbor", "dense"],
            "cannot be combined",
        ),
        ([], "--func"),
    ],
    ids=[
        "functional",
        "variant",
        "cutoff",
        "width",
        "max-triples",
        "nlist-grad",
        "compile-recompute",
        "nlist-dense",
        "no-func",
    ],
)
def test_bad_arguments_are_usage_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    """Each bad argument ends the run with a usage error, not a
    traceback."""
    path = tmp_path / "water.xyz"
    path.write_text(_WATER)

    with pytest.raises(SystemExit) as exc:
        main([*argv, str(path)])

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_cuda_without_a_cuda_device_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "water.xyz"
    path.write_text(_WATER)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="no CUDA device"):
        main(["--cuda", "--func", "pbe0", str(path)])


def test_cuda_run_moves_the_structure_to_the_device(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The move is its own timed step and the parameters are loaded on the
    device. There is no device here, so both are recorded and left
    undone."""
    path = tmp_path / "water.xyz"
    path.write_text(_WATER)
    moved_to: list[torch.device] = []
    loaded_on: list[torch.device | None] = []
    from_functional = d3.DampingParam.from_functional

    def record_move(self: Structure, device: torch.device) -> Structure:
        moved_to.append(device)
        return self

    def record_load(*args: object, **kwargs: object) -> d3.DampingParam:
        loaded_on.append(kwargs.pop("device"))  # type: ignore[arg-type]
        return from_functional(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(Structure, "to", record_move)
    monkeypatch.setattr(d3.DampingParam, "from_functional", record_load)

    argv = ["--timing", "--cuda", "--neighbor", "dense", "--func", "pbe0"]
    assert main([*argv, str(path)]) == 0

    assert moved_to == [torch.device("cuda")]
    assert loaded_on == [torch.device("cuda")]
    assert "move to device" in capsys.readouterr().out.split("Timing")[-1]


def test_module_runs_as_a_script(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python -m tad_dftd3` is the command line tool."""
    monkeypatch.setattr(sys, "argv", ["tad_dftd3", "--help"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("tad_dftd3", run_name="__main__")

    assert exc.value.code == 0
    assert "usage: tad_dftd3" in capsys.readouterr().out


@requires_compile
@pytest.mark.usefixtures("reset_dynamo")
def test_compiled_run_matches_the_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--compile` traces every step as one graph each, the three-body term
    over triples enumerated beforehand, and gives the energy and gradient
    of `dftd3`. Compiling is a timed step of its own, before the steps."""
    path = _molecule(tmp_path)
    argv = ["--func", "pbe0", "--compile", "--grad", "--timing", str(path)]

    assert main(argv) == 0
    out = capsys.readouterr().out

    energy, gradient = _reference(path, atm=True, sparse=True)
    assert _values(out, "total") == pytest.approx(energy.tolist(), abs=1e-11)
    norm = float(torch.linalg.vector_norm(gradient))
    assert _values(out, "norm") == pytest.approx([norm], abs=1e-11)

    table = out.split("Timing")[-1]
    assert table.index("triples") < table.index("compile (warm-up)")
    assert table.index("compile (warm-up)") < table.index("two-body (sparse)")
    assert "disp3" in out.split("Neighbour lists")[-1]
