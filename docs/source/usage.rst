Usage
=====

Package Usage Example
---------------------

To install the library from this repository, execute the following commands:

.. code-block:: python

    !pip install git+https://github.com/jacobmorrier/fast-er.git#egg=fast-er

    import faster

    
Here is an example of a standard probabilistic record linkage pipeline, using the previously described classes:

.. code-block:: python

    # Comparison Class
    vars = ['last_name', 'first_name', 'house_number', 'street_address']

    comp = faster.Comparison(df_A, df_B, vars, vars)

    comp.fit()

    # Estimation Class
    est = faster.Estimation(len(vars), comp.Counts)

    est.fit()

    #Linkage Class
    link = faster.Linkage(df_A, df_B, comp.Indices, est.Ksi)

    df_linked = link.transform()


Jaro-Winkler Distance Calculation Example
------------------------------------------

The following example demonstrates how to compute the Jaro-Winkler similarity between two lists of strings using CuPy:

.. code-block:: python

    import cupy as cp
    from Levenshtein import jaro_winkler

    # Example lists of strings
    strings_a = ["John Doe", "Jane Doe", "Johnny Appleseed"]
    strings_b = ["Jon Doe", "Janet Doe", "John Appleseed"]

    # Convert lists to CuPy arrays
    strings_a_cp = cp.array(strings_a)
    strings_b_cp = cp.array(strings_b)

    # Compute Jaro-Winkler similarity for all pairs of strings
    def compute_jaro_winkler(s1, s2):
        return jaro_winkler(s1, s2)

    similarities = cp.array([
        [compute_jaro_winkler(a, b) for b in strings_b_cp]
        for a in strings_a_cp
    ])

    print(similarities)

