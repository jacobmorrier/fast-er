Usage Example
=============

Here is an example of a standard probabilistic record linkage pipeline, using the previously described classes:

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
    Fast-ER employs aggressive GPU memory management to optimize the utilization of your GPU resources, but the size of the datasets you can join or deduplicate remains constrained by your GPU's memory capacity.

    The default parameters are optimized for a 16 GB Tesla T4, such as the one on Google Colab. Depending on your hardware, you may need to adjust these settings for optimal performance.

    For example, Fast-ER processes the Jaro-Winkler similarity metric in chunks to stay within a predefined memory limit. Using larger chunks typically enhances performance, as each chunk has a fixed overhead. If you have more GPU memory, consider increasing the chunk size using the ``Max_Chunk_Size`` parameter in the :doc:`Comparison <comparison>` and :doc:`Deduplication <deduplication>` classes.

Blocking
--------

Blocking consists of restricting comparisons to pairs with identical values for certain variables. For example, you may want to compare only observations with the same gender. This can be achieved by executing the doc:`Comparison <comparison>` or :doc:`Deduplication <deduplication>` class on each block distinctly. The counts for each block can then be summed elementwise to estimate the parameters of the Fellegi-Sunter model.

.. code-block:: python

    # Comparison Class
    vars = ['last_name', 'first_name', 'house_number', 'street_address']

    mComp = faster.Comparison(df_A.loc[df_A.Gender == 'M'].reset_index(), df_B.loc[df_B.Gender == 'M'].reset_index(), vars, vars, ['birth_year'], ['birth_year'])

    mComp.fit()

    fComp = faster.Comparison(df_A.loc[df_A.Gender == 'F'].reset_index(), df_B.loc[df_B.Gender == 'F'].reset_index(), vars, vars, ['birth_year'], ['birth_year'])

    fComp.fit()

    # Estimation Class
    est = faster.Estimation(len(vars), 1, mComp.Counts + fComp.Counts)

    est.fit()

    #Linkage Class
    mLink = faster.Linkage(df_A.loc[df_A.Gender == 'M'].reset_index(), df_B.loc[df_A.Gender == 'M'].reset_index(), mComp.Indices, est.Ksi)

    mdf_linked = mLink.transform()

    fLink = faster.Linkage(df_A.loc[df_A.Gender == 'F'].reset_index(), df_B.loc[df_A.Gender == 'F'].reset_index(), fComp.Indices, est.Ksi)

    fdf_linked = fLink.transform()
