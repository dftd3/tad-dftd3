Torch autodiff for DFT-D3
=========================

|release|
|license|
|testubuntu|
|testmacos|
|testwindows|
|docs|
|coverage|
|precommit|

Implementation of the DFT-D3 dispersion model in PyTorch.
This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

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


Example
-------

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

    energy = d3.dftd3(numbers, positions, param)

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
    ref = d3.reference.Reference()
    rcov = d3.data.COV_D3[numbers]
    rvdw = d3.data.VDW_D3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
    r4r2 = d3.data.R4R2[numbers]
    param = {
        "a1": torch.tensor(0.49484001),
        "s8": torch.tensor(0.78981345),
        "a2": torch.tensor(5.73083694),
    }

    cn = mctc.ncoord.cn_d3(
        numbers, positions, counting_function=mctc.ncoord.exp_count, rcov=rcov
    )
    weights = d3.model.weight_references(numbers, cn, ref, d3.model.gaussian_weight)
    c6 = d3.model.atomic_c6(numbers, weights, ref)
    energy = d3.disp.dispersion(
        numbers,
        positions,
        param,
        c6,
        rvdw,
        r4r2,
        d3.disp.rational_damping,
    )

    torch.set_printoptions(precision=10)
    print(torch.sum(energy, dim=-1))
    # tensor([-0.0014092578, -0.0057840119])


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

.. |testmacos| image:: https://github.com/dftd3/tad-dftd3/actions/workflows/macos.yaml/badge.svg
   :target: https://github.com/dftd3/tad-dftd3/actions/workflows/macos.yaml
   :alt: Tests macOS

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
