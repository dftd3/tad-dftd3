Installation
============

Clone the repository from the GitHub `repository <https://github.com/awvwgk/tad-dftd3>`__:

.. code::

   git clone https://github.com/awvwgk/tad-dftd3
   cd tad-dftd3

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

   mamba env create -n torch -f environment.yml
   mamba activate torch

Install this project with pip in the environment

.. code::

   pip install .

Add the option ``-e`` for installing in development mode.

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `torch <https://pytorch.org/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

You can check your installation by running the test suite with

.. code::

   pytest tests/ --pyargs tad_dftd3 --doctest-modules
