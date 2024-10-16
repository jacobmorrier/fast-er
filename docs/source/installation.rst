Installation
============

To install the Fast-ER package, execute the following commands:

.. code-block:: python

   !pip install git+https://github.com/jacobmorrier/fast-er.git#egg=fast-er
    
   import faster

Technical Requirements
----------------------

The technical requirements for installing the package are determined by its dependencies:

#. `CuPy <https://docs.cupy.dev/en/stable/install.html>`_
#. `Matplotlib <https://matplotlib.org/stable/install/index.html>`_
#. `NumPy <https://numpy.org/install/>`_
#. `Pandas <https://pandas.pydata.org/docs/getting_started/install.html>`_

.. warning::
    The package requires an NVIDIA CUDA GPU with a Compute Capability of 3.0 or higher.

.. tip::
   You can access CUDA-enabled GPUs for free through `Google Colab <https://colab.research.google.com>`_. 
