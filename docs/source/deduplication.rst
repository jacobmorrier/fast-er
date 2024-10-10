Deduplication
=============

This class represents our main contribution, as it performs GPU-accelerated computation of the `Jaro-Winkler similarity <https://en.wikipedia.org/wiki/Jaro–Winkler_distance>`_ for each pair of values in a dataset.

.. currentmodule:: faster.deduplication
.. autoclass:: Deduplication

Utility Functions
-----------------

These functions are used internally by the ``Deduplication`` class. Users could use them to create their own deduplication pipelines.

.. currentmodule:: faster.deduplication
.. autofunction:: jaro_winkler_dedup_gpu

.. currentmodule:: faster.deduplication
.. autofunction:: exact_dedup_gpu
