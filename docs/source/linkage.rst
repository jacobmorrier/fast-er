Linkage
=======

This class integrates the outputs from the :doc:`Comparison <comparison>` and :doc:`Deduplication <deduplication>` classes with the parameters of the Fellegi-Sunter model estimated by the :doc:`Estimation <estimation>` class, using the latter to identify the records most likely to refer to the same unit of observation.

.. tip:: To extract duplicates in a dataset, simply provide the same dataframe as input for both the ``df_A`` and ``df_B`` arguments.

.. currentmodule:: faster.linkage
.. autoclass:: Linkage
   :members:
