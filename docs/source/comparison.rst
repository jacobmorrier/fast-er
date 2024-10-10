Comparison
==========

This class represents our main contribution, as it performs GPU-accelerated computation of the `Jaro-Winkler similarity <https://en.wikipedia.org/wiki/Jaro–Winkler_distance>`_ for each pair of values between two datasets. 

For reference, the Jaro-Winkler similarity is a continuous measure that ranges from 0 to 1. We calculate the similarity between two strings, :math:`s_1` and :math:`s_2`, using the following formula:

.. math::

    \mathcal{S}(s_1, s_2) = \mathcal{J}(s_1, s_2) + \ell \times w \times \left(1 - \mathcal{J}(s_1, s_2)\right),

where:

.. math::

    \mathcal{J}(s_1, s_2) = \frac{1}{3} \left( \frac{m}{\left|s_1\right|} + \frac{m}{\left|s_2\right|} + \frac{m-\frac{t}{2}}{m}\right).

In these equations, :math:`\left|s\right|` denotes the length of string :math:`s`, :math:`m` is the number of matching characters between the strings, and :math:`t` is the number of transpositions between matching characters. Furthermore, :math:`\ell` (ranging from 0 to 4) represents the number of consecutive matching characters at the beginning of both strings, and :math:`w` (ranging from 0 to 0.25) is the weight assigned to :math:`\ell`. We discretize the Jaro-Winkler similarity so that the values of the agreement vectors :math:`\mathbf{\gamma}` are integers between 0 and :math:`L-1`, with higher integer values reflecting a greater similarity. In practice, we categorize the Jaro-Winkler similarity into three levels, using two thresholds to define these partitions.

.. currentmodule:: faster.comparison
.. autoclass:: Comparison

Utility Functions
-----------------

These functions are used internally by the ``Comparison`` class. Users could use them to create their own linkage pipelines.

.. currentmodule:: faster.comparison
.. autofunction:: jaro_winkler_gpu

.. currentmodule:: faster.comparison
.. autofunction:: jaro_winkler_unique_gpu

.. currentmodule:: faster.comparison
.. autofunction:: exact_gpu
