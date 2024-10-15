Fast-ER: GPU-Accelerated Record Linkage in Python
=================================================

.. toctree::
   :hidden:

   installation
   architecture
   usage

.. toctree::
   :caption: API Documentation
   :hidden:

   comparison
   deduplication
   estimation
   linkage
   evaluation

.. image:: https://readthedocs.org/projects/fast-er/badge/?version=latest
    :target: https://fast-er.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**Authors:**

- `Jacob Morrier <https://www.jacobmorrier.com>`_
- `Sulekha Kishore <https://www.linkedin.com/in/sulekha-kishore/>`_
- `R. Michael Alvarez <https://www.rmichaelalvarez.com>`_

Summary
-------

Record linkage, also called "entity resolution," consists of matching observations from two datasets representing the same unit, even when consistent common identifiers are absent. This process typically involves computing string similarity metrics, such as the Jaro-Winkler metric, for all pairs of values between the datasets. The Fast-ER package accelerates these computations using graphical processing units (GPUs). It estimates the parameters of the Fellegi-Sunter model, a popular probabilistic record linkage model, and performs the necessary data preprocessing, including the computation of string similarity metrics, on CUDA-enabled GPUs. This approach increases processing speed by over 60 times, reducing processing time from hours to minutes, compared to the previous leading software implementation. This significantly improves the scalability of record linkage and deduplication for large datasets.
