Installation
============

PyPI
----

*tad-dftd3* can easily be installed with ``pip``.

.. code::

  pip install tad-dftd3


From Source
-----------

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

For development, install the following additional dependencies

.. code::

   mamba install black coverage covdefaults mypy pre-commit pylint pytest tox


Install this project with pip in the environment

.. code::

   pip install .

Add the option ``-e`` for installing in development mode.

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `torch <https://pytorch.org/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

You can check your installation by running the test suite with pytest

.. code::

   pytest tests/ --pyargs tad_dftd3

or tox for testing multiple Python versions

.. code::

  tox
