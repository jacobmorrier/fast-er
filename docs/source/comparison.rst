Comparison
==========

This class contains our primary contribution, 
as it performs the GPU-accelerated computation of the Jaro-Winkler similarity 
to compare each pair of values between two datasets.

.. code-block:: python
    
    class Comparison(df_A, df_B, vars_A, vars_B)

This class evaluates the similarity between the values in two datasets using the Jaro-Winkler metric.

Functions
=========

.. currentmodule:: faster.comparison
.. autofunction:: jaro_winkler_gpu

.. currentmodule:: faster.comparison
.. autofunction:: jaro_winkler_gpu_unique    

.. currentmodule:: faster.comparison
.. autofunction:: merge_indices_pair      

.. currentmodule:: faster.comparison
.. autofunction:: merge_indices   

.. currentmodule:: faster.comparison
.. autofunction:: merge_indices   
