Installation
------------

pip
~~~

.. image:: https://img.shields.io/pypi/v/tad-dftd3
    :target: https://pypi.org/project/tad-dftd3/
    :alt: PyPI

*tad-dftd3* can easily be installed with ``pip``.

.. code::

    pip install tad-dftd3

conda
~~~~~

.. image:: https://img.shields.io/conda/vn/conda-forge/tad-dftd3.svg
    :target: https://anaconda.org/conda-forge/tad-dftd3
    :alt: Conda Version

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
