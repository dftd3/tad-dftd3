Torch autodiff for DFT-D3
=========================

|release|
|license|
|testubuntu|
|testmacos_arm|
|testwindows|
|docs|
|coverage|
|precommit|

Implementation of the DFT-D3 dispersion model in PyTorch.
This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

If you use this software, please cite the following publication

- *J. Chem. Phys.*, **2024**, *161*, 062501. (`DOI <https://doi.org/10.1063/5.0216715>`__)

For details on the D3 dispersion model see

- *J. Chem. Phys.*, **2010**, *132*, 154104 (`DOI <https://dx.doi.org/10.1063/1.3382344>`__)
- *J. Comput. Chem.*, **2011**, *32*, 1456 (`DOI <https://dx.doi.org/10.1002/jcc.21759>`__)

For alternative implementations also check out

`simple-dftd3 <https://dftd3.readthedocs.io>`__:
  Simple reimplementation of the DFT-D3 dispersion model in Fortran with Python bindings

`torch-dftd <https://tech.preferred.jp/en/blog/oss-pytorch-dftd3/>`__:
  PyTorch implementation of DFT-D2 and DFT-D3

`dispax <https://github.com/awvwgk/dispax>`__:
  Implementation of the DFT-D3 dispersion model in Jax M.D.


Installation
------------

pip
~~~

|pypi|

The project can easily be installed with ``pip``.

.. code::

    pip install tad-dftd3

conda
~~~~~

|conda|

*tad-dftd3* is also available from ``conda``.

.. code::

    conda install tad-dftd3

From source
~~~~~~~~~~~

This project is hosted on GitHub at `dftd3/tad-dftd3 <https://github.com/dftd3/tad-dftd3>`__.
Obtain the source by cloning the repository with

.. code::

    git clone https://github.com/dftd3/tad-dftd3
    cd tad-dftd3

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

    mamba env create -n torch -f environment.yml
    mamba activate torch

Install this project with ``pip`` in the environment

.. code::

    pip install .

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `tad-mctc <https://github.com/tad-mctc/tad-mctc/>`__
- `torch <https://pytorch.org/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

Compatibility
~~~~~~~~~~~~~

Python 3.10 or newer and PyTorch 2.4 or newer are required.
Older releases are not supported.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - PyTorch / Python
     - 3.10
     - 3.11
     - 3.12
     - 3.13
     - 3.14
   * - 2.4.1
     - ✔️
     - ✔️
     - ✅
     - ❌
     - ❌
   * - 2.5.1
     - ✔️
     - ✔️
     - ✅
     - ❌
     - ❌
   * - 2.6.0
     - ✔️
     - ✔️
     - ✔️
     - ✅
     - ❌
   * - 2.7.1
     - ✔️
     - ✔️
     - ✔️
     - ✅
     - ❌
   * - 2.8.0
     - ✔️
     - ✔️
     - ✔️
     - ✅
     - ❌
   * - 2.9.1
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✅
   * - 2.10.0
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✅
   * - 2.11.0
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✅
   * - 2.12.1
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✅
   * - 2.13.0
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✅
   * - 2.14.0
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

Legend: ✅ tested in CI; ✔️ supported, but not tested in CI (should still work); ❌ not supported.
Only the latest bug fix version is listed, but all preceding bug fix versions of the same minor release are supported.
For example, while 2.4.1 appears in the table, 2.4.0 is supported as well.

``torch.compile`` supports Python 3.14 only from PyTorch 2.10 on; with older PyTorch releases, use Python 3.13 or older for compilation.


Development
-----------

For development, additionally install the following tools in your environment.

.. code::

    mamba install black covdefaults mypy pre-commit pylint pytest pytest-cov pytest-xdist tox
    pip install pytest-random-order

With pip, add the option ``-e`` for installing in development mode, and add ``[dev]`` for the development dependencies

.. code::

    pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

    pre-commit install

For testing all Python environments, simply run `tox`.

.. code::

    tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

.. code::

    tox -- test


Testing against the s-dftd3 reference
--------------------------------------

Some tests compare tad-dftd3's output against reference values computed at
test time by calling the `s-dftd3 <https://github.com/dftd3/s-dftd3>`__
Fortran implementation through its Python bindings, rather than against
numbers stored in this repository. The ``dftd3`` package is therefore a
required test dependency, not an optional one -- it is pulled in by the
``[dev]`` extra, and there is no offline fallback if it is missing.

The coordination number, the weights of the reference systems and the C6
coefficients have no accessor in those bindings. They are stored in
``test/references/``, generated once from a from-source build of s-dftd3 by
``tools/refs/gen_refs.py`` (see ``tools/refs/README.md``).


Element parameters
------------------

The covalent radii, the van der Waals radii and the r⁴/r² expectation
values are element constants. Wherever they can be passed (``dftd3``,
``D3Model``, the functions of ``disp``, ``damping.atm`` and ``sparse``), they
are **per-element tables** indexed by
atomic number, with entry 0 as the dummy. Each table must have the shape
of its default:

- ``rcov_table``: ``(119,)``, default ``tad_mctc.data.radii.COV_D3``
- ``r4r2_table``: ``(119,)``, default ``tad_dftd3.data.R4R2``
- ``rvdw_table``: ``(104, 104)``, default ``tad_mctc.data.radii.VDW_PAIRWISE``

A table can be given as a tensor or as a function of ``device`` and
``dtype`` that builds it. Gradients with respect to a table come out per
element, summed over all atoms and systems, which is what parameter fitting
needs. A tensor of any other shape, e.g. one value per atom, is rejected
with a ``ValueError``.


Three-body term
---------------

Whether the Axilrod-Teller-Muto term is evaluated is a static choice, made
from the type of ``s9``, never from its value as a tensor: by default (no
``damping`` given) it is skipped if ``param`` has no ``"s9"`` or ``s9`` is
the Python number zero. A tensor ``s9`` always gets the term, even at zero,
since ``torch.compile`` and ``vmap`` cannot branch on its value and its
derivative with respect to ``s9`` is the three-body energy. Its cost grows
with the cube of the number of atoms, so to skip it, give ``s9`` as a Python
number or leave it out:

.. code:: python

    param = {"a1": a1, "a2": a2, "s8": s8, "s9": 0.0}  # no three-body term
    energy = torch.compile(d3.dftd3)(structure, param)

The parameter loader in ``tad_dftd3.param`` returns tensors, so overwrite
``param["s9"]`` with a Python number (or delete it) to get the same effect.


Periodic systems
----------------

Give the ``Structure`` lattice vectors (as rows, in Bohr) to treat the system
as a periodic cell. The coordination number and the two-body energy then sum over all
periodic images within their cutoffs, as in s-dftd3. ``periodic`` (default: all
three axes) selects the periodic axes of a slab or chain, and atoms need not
lie inside the cell. Forces and the stress follow from one backward pass:

.. code:: python

    import torch
    import tad_dftd3 as d3
    import tad_mctc as mctc

    # diamond, conventional cubic cell
    numbers = mctc.convert.symbol_to_number(symbols=8 * ["C"])
    fractional = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50],
            [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
        dtype=torch.double,
    )
    lattice = 6.7406 * torch.eye(3, dtype=torch.double)
    positions = (fractional @ lattice).requires_grad_(True)
    lattice.requires_grad_(True)
    param = {  # PBE-D3(BJ)
        "a1": torch.tensor(0.4289, dtype=torch.double),
        "s8": torch.tensor(0.7875, dtype=torch.double),
        "a2": torch.tensor(4.4407, dtype=torch.double),
    }

    structure = mctc.Structure(numbers=numbers, positions=positions, lattice=lattice)
    energy = d3.dftd3(structure, param)
    print(f"{energy.sum():.10f}")
    # -0.0562022045

    grad_pos, grad_lat = torch.autograd.grad(energy.sum(), (positions, lattice))
    virial = positions.mT @ grad_pos + lattice.mT @ grad_lat
    stress = virial / torch.linalg.det(lattice)

Which images lie within a cutoff depends on the values of the lattice, so by
default the image shifts are built anew on each call. Under ``torch.compile``
(``fullgraph=True``), ``vmap`` over cells, or ``jacrev``/``jacfwd`` with
respect to the lattice, build them once beforehand, at the larger of the
coordination-number and two-body cutoffs, and pass them as ``shifts``:

.. code:: python

    from tad_mctc.neighbor.images import build_periodic_shifts

    cutoff = d3.Cutoff()
    shifts = build_periodic_shifts(
        structure.lattice.detach(), structure.periodic, max(cutoff.cn, cutoff.disp2)
    )

    def total(pos, lat):
        cell = structure.replace(positions=pos, lattice=lat)
        return d3.dftd3(cell, param, shifts=shifts).sum()

    compiled = torch.compile(total, fullgraph=True)
    grad_lat = torch.func.jacrev(total, argnums=1)(positions, lattice)

For a batch of cells, build one table from the stacked lattices; it covers
every cell of the batch. Under ``vmap`` over a batched ``Structure``, every
field is split per system, so give each system its own lattice and periodic
mask (e.g. ``lattice.expand(nbatch, 3, 3)``), not one shared by the batch,
which fails with an ``IndexError``. Whether a table covers the lattice is only checked in
eager mode, not under ``torch.compile``, ``vmap`` or ``jacrev``, where a table
built for a larger cell silently misses images. If the cell shrinks (cell
relaxation, NPT), rebuild the table, or call
``shifts.check_compatible(structure, cutoff)`` once eagerly beforehand.

By default the evaluation is dense: every atom is paired with every image of
every atom, which takes memory proportional to ``n_atoms**2 * n_images``. For
larger systems, use neighbour lists (see `Neighbour lists`_). The three-body
term of a cell sums, as in s-dftd3, over the triples of an atom of the cell
with two atoms or images within ``Cutoff.disp3`` of it. Its dense evaluation
takes ``n_images**2`` triples per atom, which is large for a small cell at the
default cutoff of 40 Bohr: pass a smaller ``Cutoff.disp3``, or an
``nbl_disp3``. A shift table passed as ``shifts`` must cover the cutoff of
every term that has no neighbour list, so also ``Cutoff.disp3`` if the
three-body term is used.


Neighbour lists
---------------

The coordination number, the two-body and the three-body energy can be summed
over a padded, pre-built neighbour list (:class:`tad_mctc.neighbor.list.NeighborList`)
instead of all pairs. The memory then grows linearly with the number of atoms
(for a fixed cutoff), for molecules, batches and periodic cells alike, and the
energy and its derivatives to any order are those of the dense evaluation:

.. code-block:: python

    energy = d3.dftd3(structure, param, sparse=True)

``sparse=True`` builds one list per cutoff eagerly, with a single shared
search: of the coordination number, the two-body term and, if the damping has
one, the three-body term. The triples of the last grow as the sixth power of
its cutoff, so for a large system a ``disp3`` below the default of 40 Bohr may
be needed (the dense three-body term of a molecule takes memory cubic in the
number of atoms, whatever the cutoff). To reuse the lists, e.g. over the steps
of a molecular dynamics, build them once with ``build_neighborlists`` and pass
them as ``nbl_cn``, ``nbl_disp2`` and ``nbl_disp3``. A list built at a larger
cutoff, or with a ``skin``, gives the same energy, as long as the atoms have
not moved too far for the skin: rebuild the lists when ``stale`` says so.
The three-body term runs over the triangles of its list, each holding its
three sides as pairs of the list; built once with the list
(``d3.sparse.TripleList``) and passed in its place, they are reused as well,
which in a molecular dynamics of 3000 atoms took half the time per step of
building them on each call:

.. code-block:: python

    from tad_mctc.neighbor.list import build_neighborlists

    cutoff = d3.Cutoff()
    cutoffs = (cutoff.cn, cutoff.disp2, cutoff.disp3)


    def build(structure):
        nbl_cn, nbl_disp2, nbl_disp3 = build_neighborlists(
            structure, cutoffs, skin=1.0
        )
        triples = d3.sparse.TripleList.from_neighborlist(nbl_disp3)
        return dict(nbl_cn=nbl_cn, nbl_disp2=nbl_disp2, nbl_disp3=triples)


    lists = build(structure)
    energy = d3.dftd3(structure, param, cutoff=cutoff, **lists)

    # later, after the atoms have moved
    if lists["nbl_cn"].stale(structure):
        lists = build(structure)

Under autograd, ``checkpoint=True`` recomputes each chunk of pairs and of
triples in the backward pass instead of keeping it, and ``max_triples``
bounds the triples of a chunk.

A list given for a term (or built by ``sparse=True``) takes the place of
``shifts`` for that term, so ``shifts`` and lists can be combined, e.g. lists
for the pair terms and ``shifts`` for the dense three-body term of a cell. The
single-term functions take one ``pairs`` argument instead, a
``PeriodicShifts``, a ``NeighborList`` or ``None``. Over a list, pass the C6
coefficients factored per atom (``d3.model.AtomicC6``): the ``(nat, nat)``
matrix also works, but its gradient is built at full size for every chunk of
pairs, which makes the backward pass cubic in the number of atoms:

.. code-block:: python

    c6 = d3.disp.D3Model().factored_c6(structure, nbl_cn)
    d3.disp.dispersion2(structure, param, c6, pairs=nbl_disp2)

The neighbour-list evaluation itself lives in ``tad_dftd3.sparse``
(``dispersion2_sparse`` and ``dispersion_atm_sparse``), apart from the dense
one in ``tad_dftd3.disp`` and ``tad_dftd3.damping.atm``.


Smooth cutoffs
--------------

The two-body and three-body terms are cut off abruptly at ``Cutoff.disp2`` and
``Cutoff.disp3``, which makes the energy jump where a pair crosses the cutoff.
As in s-dftd3, a width switches them off smoothly instead: between
``disp - width`` and ``disp`` a contribution is scaled by a quintic that goes
from 1 to 0 with vanishing first and second derivatives. The three-body term
is scaled by the switch of each of the three distances of a triple.

.. code-block:: python

    cutoff = d3.Cutoff(disp2=60.0, width2=10.0, disp3=40.0, width3=5.0)
    energy = d3.dftd3(structure, param, cutoff=cutoff)

The widths default to zero, the hard cutoff, as in s-dftd3. There is no width
for the coordination number.


Migrating from 0.7.0
--------------------

- **Structure.** ``dftd3``, ``dispersion``, ``dispersion2``, ``dispersion3``
  and ``damping.dispersion_atm`` take a ``tad_mctc.Structure`` instead of
  ``numbers`` and ``positions``. A periodic cell is a ``Structure`` with a
  ``lattice``. To differentiate with ``torch.func``, replace the positions (or the
  lattice) inside the function with ``structure.replace(positions=...)``.

  .. code:: python

      # 0.7.0
      energy = d3.dftd3(numbers, positions, param)
      # now
      structure = mctc.Structure(numbers=numbers, positions=positions)
      energy = d3.dftd3(structure, param)

- **Element parameters.** Up to 0.7.0, ``rcov``, ``rvdw`` and ``r4r2`` took
  per-atom (or per-pair) values, i.e. ``table[numbers]``. They are now
  per-element tables, passed as ``rcov_table``, ``rvdw_table`` and
  ``r4r2_table``. The old names are unknown keyword arguments, so old calls
  fail instead of silently indexing per-atom values a second time.

  .. code:: python

      # 0.7.0
      d3.dftd3(numbers, positions, param, r4r2=d3.data.R4R2()[numbers])
      # now
      d3.dftd3(structure, param, r4r2_table=d3.data.R4R2())

- **Keyword-only arguments.** All arguments of ``dispersion``,
  ``dispersion2`` and ``dispersion3`` after ``c6``, and of
  ``damping.dispersion_atm`` after ``c6``, are keyword-only. ``dispersion``,
  ``dispersion2`` and ``dispersion3`` all take ``cutoff`` as a ``Cutoff``
  (``dispersion2`` uses its ``disp2`` and ``width2``, ``dispersion3`` its
  ``disp3`` and ``width3``); their ``width`` argument is gone. Only the
  low-level kernels (``damping.dispersion_atm`` and friends,
  ``sparse.dispersion2_sparse``) keep plain float ``cutoff`` and ``width``.

  ``dispersion2`` and ``dispersion3`` take how their pairs are enumerated as
  one ``pairs`` argument (``PeriodicShifts``, ``NeighborList`` or ``None``);
  ``dftd3`` and ``dispersion`` keep ``shifts`` and one list per term
  (``nbl_cn``, ``nbl_disp2``, ``nbl_disp3``). ``dispersion2`` takes a
  ``TwoBodyDamping`` and ``dispersion3`` a ``ThreeBodyDamping``.

  .. code:: python

      # before
      d3.disp.dispersion2(structure, param, c6, cutoff=50.0, width=5.0, nbl=nbl)
      # now
      cutoff = Cutoff(disp2=50.0, width2=5.0)
      d3.disp.dispersion2(structure, param, c6, cutoff=cutoff, pairs=nbl)

- **Three-body functions.** ``damping.dispersion_atm`` no longer dispatches:
  it is the dense term of a molecule only. A cell is
  ``damping.dispersion_atm_periodic(structure, c6, param, shifts)``, a
  neighbour list ``sparse.dispersion_atm_sparse(structure, c6, param, nbl)``;
  ``max_triples`` and ``checkpoint`` are arguments of those two (and of
  ``dispersion3``). All three take the ``param`` and a ``damping`` (a
  ``ThreeBodyDamping``, ``ZeroThreeBodyD3()`` by default) instead of ``s9``,
  ``rs9`` and ``alp``. Like the two-body damping and like dftd's
  ``get_3b_damp``, the three-body damping is now called for every triple,
  with a ``TripleData`` (distances and the van-der-Waals radii of its
  pairs), and returns its damping factor including ``s9``; a new three-body
  damping is a subclass with a ``__call__``.

  .. code:: python

      # before
      d3.damping.dispersion_atm(structure, c6, s9=1.0, alp=14.0)
      # now
      d3.damping.dispersion_atm(structure, c6, d3.DampingParam(s9=1.0, alp=14.0))

- **Model object.** ``D3Model`` (a ``tad_mctc.tree.Node``, like
  ``ncoord.cn_d3``) holds the
  configuration of the model: the element tables, reference, ``Cutoff``, and
  the counting and weighting functions. Unlike s-dftd3's class it holds no
  structure; it is called on one, single or batched. The damping is not part
  of the model. ``d3.dftd3(structure, param, **options)`` builds one from its
  keyword arguments. A tensor table is a pytree leaf, so it can be
  differentiated or batched with ``torch.func``.

  .. code:: python

      model = d3.D3Model(cutoff=d3.Cutoff(disp2=50.0))
      energy = model(structure, param)
      c6 = model.c6(structure)
      other = model.replace(r4r2_table=my_r4r2)

- **Damping.** The damping of the two- and the three-body term are chosen
  independently, as in ``dftd``: ``d3.Damping(two, three)``, e.g.
  ``d3.Damping(d3.RationalTwoBody(), d3.ZeroThreeBodyD3())`` (the default if
  the parameters set ``s9``) or ``d3.Damping(d3.RationalTwoBody())`` for no
  three-body term. Both read one shared ``d3.DampingParam`` (a
  ``tad_mctc.tree.Node``, so a pytree for ``torch.func``) with the fields
  ``s6, s8, s9, a1, a2, rs6, rs8, rs9, alp``; a field that is ``None`` is not
  set. Each damping declares defaults for some fields and raises a
  ``ValueError`` when it reads a field that is neither set nor defaulted,
  e.g. for zero-damping parameters (no ``a1``, ``a2``) given to the rational
  damping, which used to run silently with default values. The two-body damping variants of s-dftd3 are
  ``RationalTwoBody`` (BJ, and the refit BJM), ``ZeroTwoBody``,
  ``ModifiedZeroTwoBody``, ``OptimizedPowerTwoBody``, ``CSOTwoBody`` and
  ``ZTwoBody``. From the Fortran project dftd come the two-body
  ``ScreenedTwoBody`` and ``KoideTwoBody`` and the three-body
  ``RationalThreeBody``, ``ScreenedThreeBody``, ``ZeroProductThreeBody``
  (dftd's ``zero``), ``ZeroThreeBodyD4`` (dftd's ``zero_avg``, the ATM
  damping of D4) and
  ``KoideThreeBody``. They read the damping radius ``sqrt(3 r4r2_i r4r2_j)``
  (as D4) scaled to ``a1 * rdamp + a2`` (``a1`` and ``a2`` default to 1 and
  0, as in dftd), and new fields of ``DampingParam``: ``a3``, ``a4``
  (screened), ``sxc``, ``rsxc`` (Koide's exchange-correlation screening).
  They are experimental: dftd is work in progress and there are no
  reference values for them yet. ``damping_from_name("rational",
  three_body="zero_d4")`` selects a three-body damping by name (see
  ``damping.THREE_BODY_DAMPINGS``); ``True`` is D3's ``"zero_d3"``.
  ``ZeroThreeBodyD3`` and ``ZeroThreeBodyD4`` are the same averaged zero
  damping
  (``averaged_zero_damping``), the ATM damping of s-dftd3 and of dftd4
  respectively: D3 on ``rs9 * rvdw`` with the exponent ``(alp + 2) / 3``,
  D4 on ``a1 * sqrt(3 r4r2_i r4r2_j) + a2`` with ``alp / 3`` (``alp = 16``),
  so ``three_body="zero_d4"`` gives the ATM term of D4 on the C6 of D3. The
  suffix names the convention of the damping, not the model it runs in.
  ``DampingParam.from_functional(functional, damping="zero")``
  remembers the variant (``param.damping``), which ``dftd3`` uses if no
  ``damping`` is given; ``d3.damping_from_name`` builds one by name. ``rs9`` and ``alp`` moved from separate arguments
  into the parameters.

  .. code:: python

      param = d3.DampingParam.from_functional("pbe", atm=False)
      energy = d3.dftd3(structure, param)
      energy = d3.dftd3(structure, param.replace(s8=torch.tensor(0.5)))
      damping = d3.Damping(d3.RationalTwoBody(), d3.ZeroThreeBodyD3())
      energy = d3.dftd3(structure, param.replace(s9=1.0), damping=damping)

  The ``damping_function`` argument of ``dftd3``, ``dispersion`` and
  ``dispersion2`` is replaced by ``damping``, and the ``**kwargs`` that were
  passed to it are gone. A dictionary of parameters is still accepted and
  unpacked into a ``DampingParam`` as it is: nothing is filled in, so the
  rational damping needs ``s8``, ``a1`` and ``a2`` (before, they defaulted to
  1.0, 0.4 and 5.0), and an unknown key raises a ``TypeError`` (before, it
  was ignored). A missing ``s9`` means *no* three-body term, and so does a
  Python ``0.0``; a tensor ``s9`` always has it (before, a tensor zero was
  skipped in eager mode).

- **Cutoffs.** ``cutoff`` is a ``Cutoff``, not ``None`` or a single number.
  The fields of ``Cutoff`` are plain floats, not tensors.
  Any real number is accepted, including NumPy scalars; tensors are
  rejected. ``Cutoff`` no longer takes ``device`` or ``dtype`` and has no
  ``to`` method.

- **Coordination number.** ``tad_dftd3.ncoord.coordination_number`` is gone,
  since tad-mctc 0.9.0 dropped it. Use the ``cn_d3`` model instead, which
  takes the covalent radii as a table and a ``Structure``. Note that its
  default cutoff is 25 Bohr, while ``dftd3`` uses ``Cutoff.cn``
  (``d3.defaults.D3_CN_CUTOFF``, 40 Bohr):

  .. code:: python

      # 0.7.0
      cn = d3.ncoord.coordination_number(
          numbers, positions, counting_function=d3.ncoord.exp_count, rcov=rcov[numbers]
      )
      # now
      cn_model = d3.ncoord.cn_d3.replace(
          count=d3.ncoord.exp_count, rcov=rcov, cutoff=d3.defaults.D3_CN_CUTOFF
      )
      cn = cn_model(mctc.Structure(numbers=numbers, positions=positions))


Examples
--------

All examples can also be found in the `examples directory <examples>`__.

- `single.py <examples/single.py>`__
- `batch.py <examples/batch.py>`__
- `forces.py <examples/forces.py>`__
- `hessian.py <examples/hessian.py>`__

The following example shows how to calculate the DFT-D3 dispersion energy for a single structure.

.. code:: python

    import torch
    import tad_dftd3 as d3
    import tad_mctc as mctc

    numbers = mctc.convert.symbol_to_number(symbols="C C C C N C S H H H H H".split())
    positions = torch.tensor(
        [
            [-2.56745685564671, -0.02509985979910, 0.00000000000000],
            [-1.39177582455797, +2.27696188880014, 0.00000000000000],
            [+1.27784995624894, +2.45107479759386, 0.00000000000000],
            [+2.62801937615793, +0.25927727028120, 0.00000000000000],
            [+1.41097033661123, -1.99890996077412, 0.00000000000000],
            [-1.17186102298849, -2.34220576284180, 0.00000000000000],
            [-2.39505990368378, -5.22635838332362, 0.00000000000000],
            [+2.41961980455457, -3.62158019253045, 0.00000000000000],
            [-2.51744374846065, +3.98181713686746, 0.00000000000000],
            [+2.24269048384775, +4.24389473203647, 0.00000000000000],
            [+4.66488984573956, +0.17907568006409, 0.00000000000000],
            [-4.60044244782237, -0.17794734637413, 0.00000000000000],
        ]
    )
    param = {
        "a1": torch.tensor(0.49484001),
        "s8": torch.tensor(0.78981345),
        "a2": torch.tensor(5.73083694),
    }

    structure = mctc.Structure(numbers=numbers, positions=positions)
    energy = d3.dftd3(structure, param)

    torch.set_printoptions(precision=10)
    print(energy)
    # tensor([-0.0004075971, -0.0003940886, -0.0003817684, -0.0003949536,
    #         -0.0003577212, -0.0004110279, -0.0005385976, -0.0001808242,
    #         -0.0001563670, -0.0001503394, -0.0001577045, -0.0001764488])


The next example shows the calculation of dispersion energies for a batch of structures, while retaining access to all intermediates used for calculating the dispersion energy.

.. code:: python

    import torch
    import tad_dftd3 as d3
    import tad_mctc as mctc

    sample1 = dict(
        numbers=mctc.convert.symbol_to_number("Pb H H H H Bi H H H".split()),
        positions=torch.tensor(
            [
                [-0.00000020988889, -4.98043478877778, +0.00000000000000],
                [+3.06964045311111, -6.06324400177778, +0.00000000000000],
                [-1.53482054188889, -6.06324400177778, -2.65838526500000],
                [-1.53482054188889, -6.06324400177778, +2.65838526500000],
                [-0.00000020988889, -1.72196703577778, +0.00000000000000],
                [-0.00000020988889, +4.77334244722222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, -2.35039772300000],
                [-2.71400388988889, +6.70626379422222, +0.00000000000000],
                [+1.35700257511111, +6.70626379422222, +2.35039772300000],
            ]
        ),
    )
    sample2 = dict(
        numbers=mctc.convert.symbol_to_number(
            "C C C C C C I H H H H H S H C H H H".split(" ")
        ),
        positions=torch.tensor(
            [
                [-1.42754169820131, -1.50508961850828, -1.93430551124333],
                [+1.19860572924150, -1.66299114873979, -2.03189643761298],
                [+2.65876001301880, +0.37736955363609, -1.23426391650599],
                [+1.50963368042358, +2.57230374419743, -0.34128058818180],
                [-1.12092277855371, +2.71045691257517, -0.25246348639234],
                [-2.60071517756218, +0.67879949508239, -1.04550707592673],
                [-2.86169588073340, +5.99660765711210, +1.08394899986031],
                [+2.09930989272956, -3.36144811062374, -2.72237695164263],
                [+2.64405246349916, +4.15317840474646, +0.27856972788526],
                [+4.69864865613751, +0.26922271535391, -1.30274048619151],
                [-4.63786461351839, +0.79856258572808, -0.96906659938432],
                [-2.57447518692275, -3.08132039046931, -2.54875517521577],
                [-5.88211879210329, 11.88491819358157, +2.31866455902233],
                [-8.18022701418703, 10.95619984550779, +1.83940856333092],
                [-5.08172874482867, 12.66714386256482, -0.92419491629867],
                [-3.18311711399702, 13.44626574330220, -0.86977613647871],
                [-5.07177399637298, 10.99164969235585, -2.10739192258756],
                [-6.35955320518616, 14.08073002965080, -1.68204314084441],
            ]
        ),
    )
    numbers = mctc.batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = mctc.batch.pack(
        (
            sample1["positions"],
            sample2["positions"],
        )
    )
    ref = d3.reference.Reference.load()
    # per-element tables, indexed by atomic number (not per atom)
    rvdw = mctc.data.VDW_PAIRWISE()
    r4r2 = d3.data.R4R2()
    param = {
        "a1": torch.tensor(0.49484001),
        "s8": torch.tensor(0.78981345),
        "a2": torch.tensor(5.73083694),
    }

    structure = mctc.Structure(numbers=numbers, positions=positions)
    # the coordination number cutoff of `dftd3` (tad-mctc defaults to 25 Bohr)
    cn_model = d3.ncoord.cn_d3.replace(cutoff=d3.defaults.D3_CN_CUTOFF)
    cn = cn_model(structure)
    weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_log_weight)
    c6 = d3.model.atomic_c6(numbers, weights, ref)
    energy = d3.disp.dispersion(
        structure,
        param,
        c6,
        rvdw_table=rvdw,
        r4r2_table=r4r2,
    )

    torch.set_printoptions(precision=10)
    print(torch.sum(energy, dim=-1))
    # tensor([-0.0014092580, -0.0057840119])


Since the dispersion energy is differentiable with respect to the atomic
positions, the D3 contribution to the gradient (and hence the forces) is
obtained from a simple backward pass.

.. code:: python

    import torch
    import tad_dftd3 as d3
    import tad_mctc as mctc

    numbers = mctc.convert.symbol_to_number(symbols="C C C C N C S H H H H H".split())
    positions = torch.tensor(
        [
            [-2.56745685564671, -0.02509985979910, 0.00000000000000],
            [-1.39177582455797, +2.27696188880014, 0.00000000000000],
            [+1.27784995624894, +2.45107479759386, 0.00000000000000],
            [+2.62801937615793, +0.25927727028120, 0.00000000000000],
            [+1.41097033661123, -1.99890996077412, 0.00000000000000],
            [-1.17186102298849, -2.34220576284180, 0.00000000000000],
            [-2.39505990368378, -5.22635838332362, 0.00000000000000],
            [+2.41961980455457, -3.62158019253045, 0.00000000000000],
            [-2.51744374846065, +3.98181713686746, 0.00000000000000],
            [+2.24269048384775, +4.24389473203647, 0.00000000000000],
            [+4.66488984573956, +0.17907568006409, 0.00000000000000],
            [-4.60044244782237, -0.17794734637413, 0.00000000000000],
        ],
        requires_grad=True,
    )
    param = {
        "a1": torch.tensor(0.49484001),
        "s8": torch.tensor(0.78981345),
        "a2": torch.tensor(5.73083694),
    }

    structure = mctc.Structure(numbers=numbers, positions=positions)
    energy = d3.dftd3(structure, param)

    (grad,) = torch.autograd.grad(energy.sum(), positions)
    forces = -grad

The full example, including a comparison against numerical gradients, is
available in `forces.py <examples/forces.py>`__.
Second derivatives, i.e., the D3 contribution to the Hessian, are obtained by
applying reverse-mode automatic differentiation twice (see
`hessian.py <examples/hessian.py>`__).


Command line
------------

Installing the package also installs the ``tad_dftd3`` command (also run as
``python -m tad_dftd3``), which reads a structure file (xyz, Turbomole
``coord``, POSCAR, ... -- anything ``tad_mctc.io.read_structure`` reads; a
multi-frame file is a batch) and prints its dispersion energy, split into
the two- and three-body term, for the damping parameters of a functional.

.. code::

    tad_dftd3 --func pbe0 structure.xyz
    tad_dftd3 --func b3lyp --damping zero --no-atm --grad coord
    tad_dftd3 --func pbe0 --neighbor dense --cuda --omp 4 POSCAR

``--grad`` adds the gradient with respect to the positions (one backward
pass). ``--neighbor`` chooses the dense, all-pairs evaluation or the
neighbour lists (the default), and ``--nlist-only`` only builds the lists and
prints their size. The cutoffs of every term (``--cutoff-cn``,
``--cutoff-disp2``, ``--cutoff-disp3``, ``--width2``, ``--width3``) and the
memory of the backward pass (``--mode recompute``, ``--max-triples``) can be
set as in the library.

With ``--timing``, the wall time of every step is printed as it finishes
and again as a table: reading the structure, loading the native neighbour
search of tad-mctc, each step of the neighbour-list build (or the periodic
image shifts of each term), the coordination number, the reference weights,
the C6 coefficients, the two- and three-body term and the backward pass.

``--compile`` compiles every step with ``torch.compile(fullgraph=True)``.
The compilation is a timed warm-up run of its own (some seconds, again for
every new structure size), after which the steps typically run several
times faster, so it pays off for large systems. Combine it with ``--omp``
set to the number of physical cores. See ``tad_dftd3 --help`` for all
options.


Limitations
-----------

The code is fully vectorized for maximum efficiency.
Therefore, all quantities are stored as full tensors, which makes calculations rather **memory intensive**.
Especially, the ATM term can become limiting as it requires a 3D tensor of dimension ``(n_atoms, n_atoms, n_atoms)``.


Contributing
------------

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the `contributing guidelines <CONTRIBUTING.md>`__.


License
-------

Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an *“as is” basis*,
*without warranties or conditions of any kind*, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project by you, as defined in the
Apache-2.0 license, shall be licensed as above, without any additional
terms or conditions.


.. |release| image:: https://img.shields.io/github/v/release/dftd3/tad-dftd3
   :target: https://github.com/dftd3/tad-dftd3/releases/latest
   :alt: Release

.. |pypi| image:: https://img.shields.io/pypi/v/tad-dftd3
   :target: https://pypi.org/project/tad-dftd3/
   :alt: PyPI

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/tad-dftd3.svg
    :target: https://anaconda.org/conda-forge/tad-dftd3
    :alt: Conda Version

.. |license| image:: https://img.shields.io/github/license/dftd3/tad-dftd3
   :target: LICENSE
   :alt: Apache-2.0

.. |testubuntu| image:: https://github.com/dftd3/tad-dftd3/actions/workflows/ubuntu.yaml/badge.svg
   :target: https://github.com/dftd3/tad-dftd3/actions/workflows/ubuntu.yaml
   :alt: Tests Ubuntu

.. |testmacos_arm| image:: https://github.com/dftd3/tad-dftd3/actions/workflows/macos-arm.yaml/badge.svg
   :target: https://github.com/dftd3/tad-dftd3/actions/workflows/macos-arm.yaml
   :alt: Tests macOS (ARM)

.. |testwindows| image:: https://github.com/dftd3/tad-dftd3/actions/workflows/windows.yaml/badge.svg
   :target: https://github.com/dftd3/tad-dftd3/actions/workflows/windows.yaml
   :alt: Tests Windows

.. |docs| image:: https://readthedocs.org/projects/tad-dftd3/badge/?version=latest
   :target: https://tad-dftd3.readthedocs.io
   :alt: Documentation Status

.. |coverage| image:: https://codecov.io/gh/dftd3/tad-dftd3/branch/main/graph/badge.svg?token=D3rMNnl26t
   :target: https://codecov.io/gh/dftd3/tad-dftd3
   :alt: Coverage

.. |precommit| image:: https://results.pre-commit.ci/badge/github/dftd3/tad-dftd3/main.svg
   :target: https://results.pre-commit.ci/latest/github/dftd3/tad-dftd3/main
   :alt: pre-commit.ci status
