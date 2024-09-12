Usage
=====

.. _installation:

Code Example
------------

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

