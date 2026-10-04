# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. (`CLAUDE.md` in this repo is just `@AGENTS.md`, so this is the file that is actually loaded.)

## Project overview

`tad_dftd3` is a PyTorch (re-)implementation of the DFT-D3 dispersion
correction, differentiable end-to-end for single structures or batches. It
depends on the sibling package `tad-mctc` for tensor utilities, coordination
numbers, batching, and data — the public functions here
take a `tad_mctc.io.structure.Structure` (atomic numbers and coordinates in
Bohr), single or batched.

Molecules, batches and periodic cells are supported (a `Structure` with a
`lattice`); the three-body (ATM) term is molecular only and raises for a
cell (`_checks.py`). Each of the coordination number and the two-body term
has a dense path (all pairs, and for a cell every periodic image from
`PeriodicShifts`) and a sparse path over a pre-built
`tad_mctc.neighbor.list.NeighborList` (`dftd3(..., sparse=True)`, or
`nbl_cn`/`nbl_disp2`; `nbl` on `dispersion2`). The sparse ATM term takes
`nbl_disp3` and is never auto-built by `sparse=True`: its memory grows as
`cutoff**6`, so choosing it and a matching `Cutoff.disp3` is the caller's
decision. Sparse and dense must agree; `test/test_neighborlist/` checks
that, split into sparse molecular / periodic / ATM like tad-mctc's tests.

## Commands

### Environment

Create a conda env from `environment.yaml`, then install this project with
`pip install -e .[dev]` (pulls in `tad-mctc`, black, covdefaults, the `dftd3`
Python bindings, mypy, pre-commit, pylint, pytest, pytest-cov,
pytest-random-order, pytest-xdist, tox).

### Tests

```sh
pytest -q

# a single test file / test
pytest test/test_disp/test_dftd3.py
pytest test/test_disp/test_dftd3.py::test_name -vv

# by marker or keyword
pytest -m grad           # gradcheck tests (slow)
pytest -m large           # large-molecule tests (slow)
pytest -k weight_references -vv

# full matrix across supported Python/PyTorch pairs
tox                       # random order, skips "large" tests
tox -- test               # include "large" tests too
```

`pytest.ini_options` sets `addopts = --doctest-modules` — every `>>>` example
in a module docstring is executed as a test and must match its printed output
exactly (see `python-style` skill). `conftest.py` seeds `numpy`/`torch` RNGs
and enables `torch.use_deterministic_algorithms(True)` by default; `--cuda`,
`--slow`, `--detect-anomaly`/`--da`, and `--jit` toggle GPU/slow-gradcheck/
anomaly-detection/JIT modes.

Never finish a task with fewer tests passing than before you started; check
the pass/skip/xfail counts against a run before your change.

### Type checking, linting, coverage

```sh
mypy                                    # disallow_untyped_defs, strict flags
pre-commit run --files <files touched>  # isort, black (line length 80),
                                         # pyupgrade, setup-cfg-fmt, zizmor,
                                         # mypy (skipped in pre-commit.ci)
```

Coverage gate is 90% (`covdefaults` + `fail_under = 90`).

### Testing against s-dftd3

Rather than hardcoded reference tensors, tests compare against values
computed live by the Fortran `s-dftd3` implementation through its Python
bindings. `dftd3` (the PyPI package) is a **required** test dependency — it
is pulled in by the `[dev]`/`[tox]` extras — and ships a self-contained wheel
with the compiled library bundled, so no from-source build or
`LD_LIBRARY_PATH` is needed. When a test needs a new reference quantity,
extend `test/reference.py` — it holds exactly the functions its callers use
(`reference_energy_per_atom`, `reference_pairwise`,
`reference_gradient`, `reference_hessian`), not a general-purpose framework.
`cutoff=None` (its default) leaves s-dftd3 at its own compiled-in cutoffs, so
a comparison against tad-dftd3's own defaults checks *which values* those
are; pass a `tad_dftd3.cutoff.Cutoff` to pin s-dftd3 to a specific one
instead.

Prefer extending this pattern (edit `test/reference.py` / the existing test
modules) over building a parallel benchmark or demo script.

## Architecture

Public API is re-exported from `src/tad_dftd3/__init__.py`:
`dftd3` (the top-level entry point), `Cutoff`, and the submodules `cutoff`,
`damping`, `data`, `defaults`, `disp`, `model`, `ncoord`, `reference`.

The dispersion energy is assembled as a pipeline, each stage its own module:

1. **Coordination number** — `ncoord` re-exports `tad_mctc.ncoord`'s
   `coordination_number`/`cn_d3`/`exp_count`, but with the D3-specific cutoff
   default (`defaults.D3_CN_CUTOFF`) rather than `tad_mctc`'s own generic one
   — pass `cutoff` explicitly to reproduce `disp.dftd3`'s behavior if calling
   `tad_mctc.ncoord` directly.
2. **Gaussian weighting of D3 references** — `model.weight_references`, using
   `reference.Reference` (bundled reference C6/CN data, backed by
   `reference-c6.pt`) and a weighting function such as
   `model.gaussian_weight`.
3. **Atomic C6** — `model.atomic_c6` combines the weights with the reference
   table.
4. **Damping** — `damping.rational` (two-body, BJ/rational damping) and
   `damping.atm` (three-body Axilrod-Teller-Muto term, opt-in) implement the
   distance-dependent damping functions applied to the raw C6/C8 terms.
5. **Assembly** — `disp.dispersion` combines C6, van-der-Waals radii
   (`tad_mctc.data`), `r4r2` scaling (`data.r4r2`), the chosen damping
   function, and per-term cutoffs (`cutoff.Cutoff`) into the final
   atom-resolved energy; `disp.dftd3` is the convenience wrapper that also
   computes the coordination number and C6 coefficients internally.

`cutoff.Cutoff` holds one real-space cutoff per term (`cn`, `disp2`, `disp3`)
since they decay at different rates (exponential, R⁻⁶, R⁻⁹ respectively) —
mirrors s-dftd3's `realspace_cutoff` so both implementations discard the same
pairs/triples at the same distance.

Because the dispersion energy is fully differentiable, forces are a plain
`torch.autograd.grad` backward pass and the Hessian is two backward passes
(see `examples/forces.py`, `examples/hessian.py`); there is no analytic
gradient code path to keep in sync by hand.
