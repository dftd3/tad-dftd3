Installation
------------

pip
~~~

The project can easily be installed with ``pip``.

.. code::

    pip install tad-dftd3

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
- `tad_mctc <https://github.com/tad-mctc/tad_mctc/>`__
- `torch <https://pytorch.org/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)
