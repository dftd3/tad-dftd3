Examples
--------

Below, you find the most common usage examples for ``tad-dftd3``.


Single Molecule
~~~~~~~~~~~~~~~

Calculate the DFT-D3 energy of a single molecule.

.. literalinclude:: ../../examples/single.py
   :language: python
   :linenos:


Batched Calculations
~~~~~~~~~~~~~~~~~~~~

Multiple structures can be evaluated simultaneously by padding them to a common
shape (batch mode).

.. literalinclude:: ../../examples/batch.py
   :language: python
   :linenos:


Gradient / Forces
~~~~~~~~~~~~~~~~~

The dispersion energy is differentiable with respect to the atomic positions.
Hence, the D3 contribution to the gradient (and thus the forces) is obtained
from a simple backward pass. This example also verifies the analytical gradient
against a numerical one.

.. literalinclude:: ../../examples/forces.py
   :language: python
   :linenos:


Hessian
~~~~~~~

Applying reverse-mode automatic differentiation twice yields the D3
contribution to the Hessian matrix.

.. literalinclude:: ../../examples/hessian.py
   :language: python
   :linenos:
