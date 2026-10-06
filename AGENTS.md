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
`lattice`), including the three-body (ATM) term, which for a cell sums over
an atom of the cell with two images, like s-dftd3. Each of the coordination
number, the two-body term and the ATM term has a dense path (all pairs, and
for a cell every periodic image from `PeriodicShifts`) and a sparse path over a pre-built
`tad_mctc.neighbor.list.NeighborList` (`dftd3(..., sparse=True)`, or
`nbl_cn`/`nbl_disp2`/`nbl_disp3` on `dftd3`/`dispersion`; one `pairs` argument
on `dispersion2`/`dispersion3`). A list given for a term takes the place of
`shifts` for that term. `sparse=True` builds every missing list in one shared
search, the ATM one too if the damping has a three-body part: its triples
grow as `cutoff**6`, but the dense molecular ATM is O(nat³) whatever the
cutoff. `D3Model` passes C6 factored per atom (`model.AtomicC6`) and builds
the `(nat, nat)` matrix only for the dense terms: over a list, a lookup in a
matrix that needs a gradient creates one of its full size per chunk in the
backward pass, so the sparse kernels take `pair(i, j)` (they still accept a
matrix from a direct caller).
The sparse ATM term runs over the triangles of its list, tad-mctc's
`TripleList` (re-exported as `sparse.TripleList`): each triple holds its three
sides as slots of the list, so distance, C6 and radii are formed once per
pair, not per triple side (about 3x faster per MD step, eagerly, than
recomputing them per triple). Given a
list, the term builds them when called (data dependent); built once with
`TripleList.from_neighborlist(nbl)`, they take the list's place in
`nbl_disp3`/`pairs`, are reused while the list is not stale, and are
fixed-shape for `fullgraph`/`vmap`.
Sparse and dense must agree; `test/test_neighborlist/` checks
that, split into sparse molecular / periodic / ATM like tad-mctc's tests.

## Transform compatibility

Everything in this package must be compatible with `torch.func.vmap`,
`torch.compile(fullgraph=True)`, `torch.func.jacrev` and
`torch.func.jacfwd`, in addition to plain autograd. Concretely:

- Do not read tensor values to decide control flow or shapes (`.item()`,
  `bool(tensor)`, `.any()`, data-dependent shapes), and do not branch on
  whether a tensor is zero. Use `torch.where` and masks, or make the choice
  static: a Python number or `None` is a constant to the trace, as for `s9`
  (see `disp.default_damping`) and the unset fields of `DampingParam`.
- Keep configuration in `tad_mctc.tree.Node` values (tensors as leaves, the
  rest as static context) and do not rebuild them with `replace` inside code
  that must be traced; Dynamo cannot trace `Node.replace`, nor methods like
  `dict.items()` on class-level dicts (use tuples).
- Data-dependent work, such as building periodic image shifts or neighbour
  lists, happens eagerly outside the graph and is passed in
  (`shifts`, `nbl_*`).
- A new feature needs a test under each transform where it applies (see
  `test/test_disp/test_compile.py`, `test/test_grad/`,
  `test/test_periodic/test_compile.py`).

## Validation

Keep a check only if, without it, a wrong number would come out silently
(a per-atom array passed as an element table, shifts or a neighbour list too
short for a cutoff, a damping parameter that is not set). Where Python or
torch already raises, even less politely (wrong types, unknown keywords,
an element beyond the tables), let it raise. A check reads shapes, types and
`None`-ness only, or goes through tad-mctc's `check_compatible`, which skips
itself under transforms.

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

The `tad_dftd3` command line (`cli/`) reuses tad-mctc's private
`tad_mctc.cli` timer and sections, so it depends on the exact tad-mctc pin.
To time each stage it runs the pipeline below step by step through the same
calls as `D3Model` (`coordination_number`, `weight_references`,
`factored_c6`, `dispersion2`/`dispersion3`); a change to `D3Model.__call__`
needs the same change there. `test/test_cli/` checks its printed numbers
against `dftd3`.

The dispersion energy is assembled as a pipeline, each stage its own module:

1. **Coordination number** — `ncoord` re-exports `tad_mctc.ncoord`'s
   `coordination_number`/`cn_d3`/`exp_count`, but with the D3-specific cutoff
   default (`defaults.D3_CN_CUTOFF`) rather than `tad_mctc`'s own generic one
   — pass `cutoff` explicitly to reproduce `disp.dftd3`'s behavior if calling
   `tad_mctc.ncoord` directly.
2. **Gaussian weighting of D3 references** — `model.weight_references`, using
   `reference.Reference` (bundled reference C6/CN data, backed by
   `reference-c6.pt`) and a weighting function such as
   `model.gaussian_log_weight`, which returns log-weights so that the
   normalization is a softmax (no float64 detour, no underflow).
3. **Atomic C6** — `model.factored_c6` combines the weights with the
   reference table into per-atom factors (`AtomicC6`); `dense()` (=
   `model.atomic_c6`) is the `(nat, nat)` matrix the dense terms take.
4. **Damping** — a `damping.Damping` pairs a `TwoBodyDamping` (one class per
   s-dftd3 variant, e.g. `RationalTwoBody`, `ZeroTwoBody`) with an optional
   `ThreeBodyDamping`; both read one `DampingParam` through `value()`, which
   supplies the variant's defaults and raises for a missing parameter. Both
   are called by the kernels, per pair with a `PairData` and per triple with
   a `TripleData` (like dftd's `get_3b_damp`, `s9` included); the kernels
   own only geometry, masks and cutoffs. `ZeroThreeBodyD3` is the only
   three-body damping of s-dftd3 (dftd's `zero_avg` on D3's `rvdw` radii).
   The data tuples carry two radii per pair, and each damping reads the one
   it is defined with: `rvdw` (s-dftd3's zero, mzero, `ZeroThreeBodyD3`) or the
   model radius `rdamp = sqrt(3 r4r2_i r4r2_j)`, as in D4 (the dampings
   ported from the Fortran `dftd` project, which scale it to
   `a1 * rdamp + a2` with dftd's defaults 1 and 0). This prepares a general
   tad-dftd where each model supplies `rdamp`. dftd is work in progress: its
   ports have no reference values and Koide is untested, on purpose; tests
   must not depend on dftd output yet.
5. **Assembly** — `disp.dispersion` adds `dispersion2` and, if the damping
   has a three-body part, `dispersion3`; `disp.dftd3` (= `D3Model()(...)`)
   also computes the coordination number and C6 coefficients. Each term has
   one function per way of enumerating its pairs, with no dispatch inside
   them: the dense two-body term in `disp`, the dense ATM term in
   `damping.atm` (`dispersion_atm` for a molecule, kept as the reference
   implementation, and `dispersion_atm_periodic`), and both neighbour-list
   paths in `sparse`. `dispersion2`/`dispersion3` choose among them by the
   type of `pairs` only.

`cutoff.Cutoff` holds one real-space cutoff per term (`cn`, `disp2`, `disp3`)
since they decay at different rates (exponential, R⁻⁶, R⁻⁹ respectively) —
mirrors s-dftd3's `realspace_cutoff` so both implementations discard the same
pairs/triples at the same distance.
`width2`/`width3` switch the two- and three-body terms off smoothly over the
last `width` Bohr (`cutoff.smooth_cutoff`, s-dftd3's quintic); the default of
zero is the hard cutoff. Tests pin both to s-dftd3 through
`test/reference.py`.

Because the dispersion energy is fully differentiable, forces are a plain
`torch.autograd.grad` backward pass and the Hessian is two backward passes
(see `examples/forces.py`, `examples/hessian.py`); there is no analytic
gradient code path to keep in sync by hand.

### Regenerating test reference data

`test/references/<collection>/<record>.json` (coordination number, reference
weights, C6 fixtures, keyed by the `(collection, record)` of `get_structure`) is
generated by `tools/refs/gen_refs.py`, which drives a from-source Fortran
build of `s-dftd3` (via `fpm` or `meson`, see `tools/refs/README.md`) —
needed only because these three quantities have no accessor in the `dftd3`
Python package's C API. The structures are listed in `SAMPLE_LIST` as `(collection, record)` pairs for `tad_mctc.data.structures.get_structure`; the tests use the same pairs. Do not
hand-edit the generated JSON files. Used by `test/test_model/` (C6 and
weights are fed in from here, not from tad-dftd3's own code).
