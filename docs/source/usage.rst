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

.. hint::
    Fast-ER employs aggressive GPU memory management to optimize the utilization of your GPU resources, but the size of the datasets you can join or deduplicate remains constrained by your GPU's memory capacity.

    The default parameters are optimized for a 16 GB Tesla T4, such as the one on Google Colab. Depending on your hardware, you may need to adjust these settings for optimal performance.

    For example, Fast-ER processes the Jaro-Winkler similarity metric in chunks to stay within a predefined memory limit. Using larger chunks typically enhances performance, as each chunk has a fixed overhead. If you have more GPU memory, consider increasing the chunk size using the Max_Chunk_Size parameter in the Comparison and Deduplication classes.
