Fast-ER: GPU-Accelerated Record Linkage and Deduplication in Python
===================================================================

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

Fast-ER is a package for GPU-accelerated record linkage and deduplication in Python.

Key Features
------------

- Harnesses the power of **CUDA-enabled GPUs** to dramatically accelerate standard record linkage and deduplication pipelines
- Runs over 35 times **faster** than the leading CPU-powered software implementation
- Enhances the **scalability** of record linkage and deduplication for large datasets

What are Record Linkage and Deduplication?
------------------------------------------

Imagine you have two datasets like this:

.. image:: images/Example.svg

As reflected by the arrows, both datasets contain matching entries. However, they lack a common unique identifier over which we could execute a direct "hard" join. Additionally, inconsistencies across other variables make a reliable merge based on their values alone impossible.

`Record linkage <https://en.wikipedia.org/wiki/Record_linkage>`_ solves this issue by comparing the values across all pairs of entries from both datasets using similarity metrics. These metrics provide a continuous measure of similarity between two strings. The results are processed by a probabilistic model, which estimates the likelihood that each pair refers to the same entity. This approach helps to identify observations that refer to the same entity, even when none of their values match perfectly.

A similar approach can be used to deduplicate entries within a single dataset, where "hard" deduplication may fail due to the absence of a unique identifier and inconsistencies across variables.
