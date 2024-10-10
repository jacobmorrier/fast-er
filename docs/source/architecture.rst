Description of the Package Architecture
=======================================

The package consists of classes that are meant to serve as modules in a probabilistic record linkage pipeline, adaptable to the user's needs and requirements. Each class requires user-provided inputs and produces relevant outputs, some of which must be passed to other classes for further processing.

The following are two figures illustrating standard probabilistic record linkage and deduplication pipelines, respectively.

Probabilistic Record Linkage
----------------------------

Let us begin by describing the probabilistic record linkage pipeline. Its objective is to identify observations in two datasets that correspond to the same unit of observation based on the similarity of values of some variables.

The process begins by supplying the following inputs to the :doc:`Comparison <comparison>` class: (i) two Pandas data frames (``df_A`` and ``df_B``), (ii) variables to compare for fuzzy matching (``Vars_Fuzzy_A`` and ``Vars_Fuzzy_B``), and (iii) variables to compare for exact matching (``Vars_Exact_A`` and ``Vars_Exact_B``). The class then compares the values of all pairs of observations in both data frames and produces an array, stored in the ``Counts`` attribute, with the count of each pattern of discrete similarity levels across all variables. This array serves as the main input to the :doc:`Estimation <estimation>` class, which uses it to estimate the conditional match probability for each pattern of discrete similarity levels across all variables. In turn, along with the list of indices corresponding to each pattern, this information is the main input to the :doc:`Linkage <linkage>` class, which outputs a Pandas data frame with all pairs of observations with a conditional match probability exceeding a user-specified threshold.

Probabilistic Deduplication
---------------------------
