Comparison
==========

This class represents our main contribution, as it performs GPU-accelerated computation of the `Jaro-Winkler similarity <https://en.wikipedia.org/wiki/Jaro–Winkler_distance>`_ for each pair of values between two datasets.

.. currentmodule:: faster.comparison
.. autoclass:: Comparison

Utility Functions
-----------------

These functions are used internally by the ``Comparison`` class. Users could utilize them to create their own pipelines.

.. currentmodule:: faster.comparison
.. autofunction:: jaro_winkler_gpu

.. currentmodule:: faster.comparison
.. autofunction:: jaro_winkler_unique_gpu

.. currentmodule:: faster.comparison
.. autofunction:: exact_gpu
