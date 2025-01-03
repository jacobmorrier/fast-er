Estimation
==========

This class estimates the parameters of the Fellegi-Sunter model, a `Naive Bayes classifier <https://en.wikipedia.org/wiki/Naive_Bayes_classifier>`_ commonly used in probabilistic record linkage, with the output generated by either the :doc:`Comparison <comparison>` or :doc:`Deduplication <deduplication>` classes.

For reference, below is a comprehensive description of the Fellegi-Sunter model.

Suppose we want to join observations from two data sets, :math:`\mathcal{A}` and :math:`\mathcal{B}`, with sizes :math:`N_\mathcal{A}` and :math:`N_\mathcal{B}`, respectively. Both datasets have :math:`K` variables in common. We evaluate all possible pairwise comparisons of the values for these variables. Specifically, for each of the :math:`N_\mathcal{A} \times N_\mathcal{B}` pairs of values, we define an agreement vector of length :math:`K`, denoted :math:`\mathbf{\gamma}_{ij}`. The :math:`k^{\textrm{th}}` element of this vector indicates the discrete level of similarity for the :math:`k^{\textrm{th}}` variable between the :math:`i^{\textrm{th}}` observation from dataset :math:`\mathcal{A}` and the :math:`j^{\textrm{th}}` observation from dataset :math:`\mathcal{B}`.

The model presumes the existence of a latent variable :math:`M_{ij}`, which captures whether the pair of observations consisting of the :math:`i^{\textrm{th}}` observation from dataset :math:`\mathcal{A}` and the :math:`j^{\textrm{th}}` observation from dataset :math:`\mathcal{B}` constitutes a match. The model follows a simple finite mixture structure:

.. math::

    \gamma_{ij}(k) | M_{ij} = m \sim \textrm{Discrete}(\mathbf{\pi}_{km})

.. math::

    M_{ij} \sim \textrm{Bernoulli}(\lambda).

The vector :math:`\mathbf{\pi}_{km}`, of length :math:`L`, encodes the probability of each discrete similarity level being observed for the :math:`k^{\textrm{th}}` variable conditional on whether the pair is a match (:math:`m=1`) or not (:math:`m=0`). The parameter :math:`\lambda` denotes the overall probability of a match across all pairwise comparisons. The model's estimands are the parameters :math:`\lambda` and :math:`\mathbf{\pi}`. Once estimated, these parameters can be used to calculate the conditional match probability for all pairs of observations.

For more details on the Fellegi-Sunter model, refer to this `excellent paper <https://doi.org/10.1017/S0003055418000783>`_.

.. currentmodule:: faster.estimation
.. autoclass:: Estimation
   :members:
