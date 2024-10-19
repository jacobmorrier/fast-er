Usage Example
=============

Here is an example of a standard probabilistic record linkage pipeline using the package's classes:

.. code-block:: python

    # Comparison Class
    vars = ['last_name', 'first_name', 'house_number', 'street_address']

    comp = faster.Comparison(df_A, df_B, vars, vars, ['birth_year'], ['birth_year'])

    comp.fit()

    # Estimation Class
    est = faster.Estimation(len(vars), 1, comp.Counts)

    est.fit()

    #Linkage Class
    link = faster.Linkage(df_A, df_B, comp.Indices, est.Ksi)

    df_linked = link.transform()

.. tip::
    Fast-ER implements aggressive memory management to optimize the utilization of GPU resources. Nevertheless, the size of the datasets you can join or deduplicate remains constrained by the available GPU memory.

    The default parameters are optimized for a 16 GB Tesla T4, like the one provided by default on `Google Colab < https://colab.research.google.com>`_. Depending on your hardware, you may need to adjust these settings for optimal performance.

    For example, Fast-ER processes the Jaro-Winkler similarity metric in chunks to stay within a predefined memory limit. Using larger chunks typically improves performance, as each chunk has a fixed overhead. If you have more GPU memory available, consider increasing the chunk size using the ``Max_Chunk_Size`` parameter in the :doc:`Comparison <comparison>` and :doc:`Deduplication <deduplication>` classes.

Blocking
--------

Blocking consists of restricting comparisons and potential matches to pairs with identical values for certain variables. For instance, you may want to compare only observations with the same gender.

This can be achieved by executing the :doc:`Comparison <comparison>` or :doc:`Deduplication <deduplication>` class on each block distinctly. The counts for each block can then be summed elementwise to estimate the parameters of the Fellegi-Sunter model.

Here is an example of a probabilistic record linkage pipeline with gender blocking using the package's classes:

.. code-block:: python

    # Comparison Class
    vars = ['last_name', 'first_name', 'house_number', 'street_address']

    df_A_M = df_A.loc[df_A.Gender == 'M'].reset_index()

    df_B_M = df_B.loc[df_B.Gender == 'M'].reset_index()

    CompM = faster.Comparison(df_A_M, df_B_M, vars, vars, ['birth_year'], ['birth_year'])

    CompM.fit()

    df_A_F = df_A.loc[df_A.Gender == 'F'].reset_index()

    df_B_F = df_B.loc[df_B.Gender == 'F'].reset_index()

    CompF = faster.Comparison(df_A_F, df_B_F, vars, vars, ['birth_year'], ['birth_year'])

    CompF.fit()

    # Estimation Class
    est = faster.Estimation(len(vars), 1, CompM.Counts + CompF.Counts)

    est.fit()

    #Linkage Class
    LinkM = faster.Linkage(df_A_M, df_B_M, CompM.Indices, est.Ksi)

    df_M = LinkM.transform()

    LinkF = faster.Linkage(df_A_F, df_B_F), CompF.Indices, est.Ksi)

    df_F = LinkF.transform()
