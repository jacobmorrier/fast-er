Fast-ER: GPU-Accelerated Probabilistic Record Linkage in Python
===============================================================

.. toctree::
   :maxdepth: 2
   comparison
   deduplication
   estimation
   linkage

.. image:: https://readthedocs.org/projects/fast-er/badge/?version=latest
    :target: https://fast-er.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**Authors:**

- `Jacob Morrier <https://www.jacobmorrier.com>`_
- `Sulekha Kishore <https://www.linkedin.com/in/sulekha-kishore/>`_
- `R. Michael Alvarez <https://www.rmichaelalvarez.com>`_

.. contents:: **Table of Contents**
    :depth: 3

Introduction
------------

`Record linkage <https://en.wikipedia.org/wiki/Record_linkage>`_, also called "entity resolution," encompasses techniques for joining observations from two datasets that refer to the same unit or entity, even when the datasets do not share consistently formatted common identifiers.

Typically, record linkage involves computing string similarity metrics, such as the Jaro-Winkler metric, for all pairs of possible values across both datasets. While these calculations are simple, they become computationally expensive as the number of observations increases, causing the number of required comparisons to grow exponentially. For example, when linking observations from two datasets, each with 1,000,000 observations, adding just one more observation to either dataset results in an additional 1,000,000 comparisons. This makes record linkage prohibitively expensive to perform, even for datasets of moderate size.

To address this challenge, we propose to use graphical processing units (GPUs) to accelerate these computations. Originally designed for computer graphics and digital image processing, GPUs are also adept at performing parallel non-graphic calculations. This capability has been instrumental in advancing the field of artificial intelligence (AI). A growing number of software tools now support and facilitate the use of GPUs to accelerate data processing and other data science tasks. For instance, `RAPIDS <https://rapids.ai/>`_ is an open-source suite of libraries developed by NVIDIA, a leading GPU manufacturer, leveraging CUDA-enabled GPUs to accelerate workflows in data processing, machine learning, and graph analytics. Similarly, `CuPy <https://cupy.dev/>`_ is an open-source Python library designed to enable fast array-based numerical computations on GPUs.

Leveraging state-of-the-art GPU-accelerated computation tools, we have implemented the Fellegi-Sunter model, a widely used probabilistic record linkage model, along with the associated data processing tasks on CUDA-enabled GPUs. Our experiments demonstrate that this approach can accelerate the process by more than 60 times compared to the previous leading implementation (`fastLink <https://github.com/kosukeimai/fastLink/tree/master>`_). Importantly, this makes probabilistic record linkage methods more germane to large-size datasets. An open-source Python library accompanies this white paper.


Description of the Fellegi-Sunter Model [#]_
--------------------------------------------

Suppose we want to join observations from two data sets, :math:`\mathcal{A}` and :math:`\mathcal{B}`, with sizes :math:`N_\mathcal{A}` and :math:`N_\mathcal{B}`, respectively. Both datasets have :math:`K` variables in common. We evaluate all possible pairwise comparisons of the values for these variables. Specifically, for each of the :math:`N_\mathcal{A} \times N_\mathcal{B}` pairs of values, we define an agreement vector of length :math:`K`, denoted :math:`\mathbf{\gamma}_{ij}`. The :math:`k^{\textrm{th}}` element of this vector indicates the discrete level of similarity for the :math:`k^{\textrm{th}}` variable between the :math:`i^{\textrm{th}}` observation from dataset :math:`\mathcal{A}` and the :math:`j^{\textrm{th}}` observation from dataset :math:`\mathcal{B}`.

We use the `Jaro-Winkler similarity metric <https://en.wikipedia.org/wiki/Jaro–Winkler_distance>`_ to measure the similarity between two strings [#]_. The Jaro-Winkler similarity is a continuous measure that ranges from 0 to 1. We calculate the similarity between two strings, :math:`s_1` and :math:`s_2`, using the following formula:

.. math::

    \mathcal{S}(s_1, s_2) = \mathcal{J}(s_1, s_2) + \ell \times w \times \left(1 - \mathcal{J}(s_1, s_2)\right),

where:

.. math::

    \mathcal{J}(s_1, s_2) = \frac{1}{3} \left( \frac{m}{\left|s_1\right|} + \frac{m}{\left|s_2\right|} + \frac{m-\frac{t}{2}}{m}\right).

In these equations, :math:`\left|s\right|` denotes the length of string :math:`s`, :math:`m` is the number of matching characters between the strings, and :math:`t` is the number of transpositions between matching characters. Furthermore, :math:`\ell` (ranging from 0 to 4) represents the number of consecutive matching characters at the beginning of both strings, and :math:`w` (ranging from 0 to 0.25) is the weight assigned to :math:`\ell`. We discretize the Jaro-Winkler similarity so that the values of the agreement vectors :math:`\mathbf{\gamma}` are integers between 0 and :math:`L-1`, with higher integer values reflecting a greater similarity. In practice, we categorize the Jaro-Winkler similarity into three levels, using two thresholds to define these partitions.

The agreement vectors :math:`\mathbf{\gamma}` are used to estimate a naive Bayes latent variable model, which assigns weights to each variable based on its ability to distinguish between matches and non-matches. These weights are subsequently used to estimate the probability that two records refer to the same unit. In turn, this probability determines which observations are linked together.

Formally, the model presumes the existence of a latent variable :math:`M_{ij}`, which indicates whether the pair of observations consisting of the :math:`i^{\textrm{th}}` observation from dataset :math:`\mathcal{A}` and the :math:`j^{\textrm{th}}` observation from dataset :math:`\mathcal{B}` constitutes a match. [#]_ The model follows a simple finite mixture structure:

.. math::

    \gamma_{ij}(k) \sim \textrm{Discrete}(\mathbf{\pi}_{km})

.. math::

    M_{ij} \sim \textrm{Bernoulli}(\lambda).

The vector :math:`\mathbf{\pi}_{km}`, of length :math:`L`, represents the probability of each discrete similarity level being observed for the :math:`k^{\textrm{th}}` variable conditional on whether the pair is a match (:math:`m=1`) or not (:math:`m=0`). The parameter :math:`\lambda` denotes the overall probability of a match across all pairwise comparisons. The model's estimands are the parameters :math:`\lambda` and :math:`\mathbf{\pi}`. Once estimated, these parameters can be used to calculate the conditional match probability for all pairs of observations.

Calculating the Jaro-Winkler similarity between all pairs of values is highly amenable to parallelization because each pair is processed independently using the same instructions. Our main contribution consists in implementing this parallelization on GPUs.


Brief Description of General-Purpose Computing on Graphical Processing Units
----------------------------------------------------------------------------

GPUs are specialized electronic circuits designed in the 1970s to enhance digital image processing. Unlike central processing units (CPUs), which are optimized for the rapid execution of sequential instructions, GPUs are designed to perform thousands of operations simultaneously. This parallelism makes them highly efficient for tasks that can be broken down into many smaller, simultaneous calculations, particularly those involving graphics. Early applications of GPUs focused on geometric transformations, such as rotating and translating vertices between coordinate systems, as well as texture mapping, where they could process large numbers of pixels concurrently.

GPUs are highly effective not only for graphics but also for non-graphical calculations. They are especially well-suited for high-throughput computations that involve data parallelism, where the same operations are applied to multiple data points simultaneously. This is due to the GPU's Single Instruction, Multiple Data (SIMD) architecture. In this context, GPUs are used as stream or vector processors, leveraging the immense computational power of modern shader pipelines to execute "compute kernels." These kernels are like the instructions in a "for loop," except that rather than being executed sequentially, they are executed concurrently across data points. The immense computation power of modern GPUs allows them to achieve performance levels that can be several orders of magnitude faster than traditional CPUs, particularly for applications involving extensive vector or matrix operations.

Storage and Manipulation of Strings with the Arrow Columnar Format
------------------------------------------------------------------

The main challenge in implementing the Jaro-Winkler similarity metric on GPUs, and more broadly in working with strings, stems from the fact that they do not natively support "arrays of arrays." Since a string is essentially an array of characters, an array of strings represents an array of arrays, which complicates the task of storing strings on the GPU.

A convenient and simple solution to this problem is to flatten the arrays of characters into a single array of characters and separately store pointers that track the start and end of each string. This representation is called "columnar format" or "columnar memory layout." It is used by several libraries, such as `PyArrow <https://arrow.apache.org/docs/python/>`_ and `RAPIDS cuDF <https://docs.rapids.ai/api/cudf/stable/>`_. By arranging data in a columnar format, the task of handling arrays of strings on GPUs becomes significantly easier.

This strategy is efficient in terms of memory usage and access patterns. It eliminates the overhead associated with "jagged" arrays, where the length of the inner arrays varies. Additionally, when strings are stored in a flattened array, it becomes easier to apply GPU-friendly optimizations, such as loading large chunks of memory into cache for fast access or applying the same operation across multiple strings concurrently.

GPU-Accelerated Record Linkage with CuPy
----------------------------------------

In our GPU-accelerated implementation of the Fellegi-Sunter model, we rely heavily on the `CuPy <https://cupy.dev/>`_ library. CuPy is an open-source library for GPU-accelerated array-based numerical computations in Python. It provides an interface that is highly similar to NumPy, making it an intuitive choice for Python developers who want to leverage the computational power of GPUs without needing to learn CUDA programming.

CuPy is built on CUDA, a parallel computing platform and programming model developed by NVIDIA. By using CuPy, developers can offload array computations to the GPU with minimal code changes. CuPy supports a wide range of numerical operations, including those commonly used in scientific computing and machine learning. Moreover, CuPy has seamless interoperability with other GPU-accelerated libraries, such as RAPIDS cuDF, which we use for data manipulation and preprocessing.

Our implementation involves two main tasks: computing the Jaro-Winkler similarity for all pairs of strings and estimating the parameters of the Fellegi-Sunter model using maximum likelihood estimation (MLE). Both of these tasks are accelerated using CuPy, allowing us to leverage the parallel processing capabilities of GPUs.


Conclusion
----------

We have implemented the Fellegi-Sunter model for probabilistic record linkage using state-of-the-art GPU-accelerated computation tools. Our experiments demonstrate that this approach can accelerate the record linkage process by over 60 times compared to existing implementations, making it feasible to perform record linkage on large datasets. This acceleration is achieved by leveraging the parallel processing capabilities of GPUs through libraries such as CuPy and RAPIDS cuDF. Our implementation is open-source, and we hope it will provide a valuable resource for researchers and practitioners working with large-scale datasets.

Acknowledgments
---------------

This work was supported by funding from the National Science Foundation (NSF) and NVIDIA Corporation. We are also grateful to the contributors of the open-source libraries CuPy and RAPIDS cuDF, whose tools made this project possible.

References
----------
.. [#] Winkler, W.E. 1990. "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage." *Proceedings of the Section on Survey Research Methods*: 354-359.
.. [#] For a more detailed description and discussion of the Fellegi-Sunter model, see this `paper <https://www.cambridge.org/core/journals/american-political-science-review/article/using-a-probabilistic-model-to-assist-merging-of-largescale-administrative-records/DB2955F64A1F4E262C5B9B26C6D7552E>`_.
.. [#] Fellegi, I.P., and A.B. Sunter. 1969. "A Theory for Record Linkage." *Journal of the American Statistical Association* 64 (328): 1183-1210.
