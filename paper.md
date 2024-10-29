---
title: 'Fast-ER: GPU-Accelerated Record Linkage and Deduplication in Python'
authors:
- name: Jacob Morrier
  orcid: 0000-0002-1815-7431
  affiliation: '1'
- name: Sulekha Kishore
  affiliation: '1'
- name: R. Michael Alvarez
  orcid: 0000-0002-8113-4451
  affiliation: '1'
affiliations:
- name: Division of the Humanities and Social Sciences, California Institute of Technology, USA
  index: 1
date: 28 October 2024
bibliography: paper.bib
---

# Summary

Record linkage, also known as "entity resolution," consists of identifying matching records across multiple datasets that lack common unique identifiers. On the other hand, deduplication involves recognizing duplicate entries within a dataset in which unique identifiers may be inconsistent or missing. These techniques are fundamental tools for research across fields such as social and health sciences [e.g., @Jutte_Roos_Brownell_2011; @Ruggles_Fitch_Roberts_2018; @Kim_Schneider_Alvarez_2020; @Yoder_2020; @Kwiek_Roszka_2021].

Both tasks typically require calculating string similarity metrics for all pairs of values between datasets. The `Fast-ER` package harnesses the computational power of graphical processing units (GPUs) to accelerate this dramatically. It estimates the widely used Fellegi-Sunter probabilistic model and performs the computationally intensive preprocessing steps, including calculating string similarity metrics, on CUDA-enabled GPUs.

`Fast-ER` runs over 35 times faster than the leading CPU-powered software implementation, reducing execution time from hours to minutes. This significantly enhances the scalability of record linkage and deduplication for large datasets.

# Statement of Need

Record linkage and deduplication typically involve calculating string similarity metrics, such as the Jaro-Winkler metric [@Winkler_1990], for all pairs of values between two datasets or within a dataset. Although these calculations are simple, the number of comparisons grows exponentially with the number of observations. For instance, when linking observations from two datasets, each with 1,000,000 observations, adding just one more observation to either dataset results in an additional 1,000,000 comparisons. This makes record linkage and deduplication prohibitively expensive to perform, even for datasets of moderate size.

GPUs were developed in the 1970s to accelerate digital image processing. Unlike central processing units (CPUs), designed for the sequential execution of a single thread of instructions with minimal latency, GPUs are optimized for performing hundreds of operations simultaneously [@kirk_hwu_2017]. Early applications focused on geometric transformations and texture mapping. GPUs can also be used for non-graphical computations. They are especially well-suited for high-throughput computations that can be broken down into identical, independent calculations, such as those exhibiting data parallelism, in which the same instructions are executed individually over many data points. This stems from GPUs’ Single Instruction, Multiple Data (SIMD) architecture. Concretely, the shader pipelines of modern GPUs can execute "compute kernels," analogous to instructions in a "for loop." However, rather than running sequentially, these operations are executed simultaneously across inputs. As a result, GPUs can often deliver performance orders of magnitude faster than traditional CPUs.

Our GPU-accelerated implementation of record linkage and deduplication relies heavily on the `CuPy` library, an open-source library for array-based numerical computations on GPUs in Python [@cupy_learningsys2017]. Built on NVIDIA's `CUDA` parallel computing model, `CuPy` has an intuitive application programming interface (API) closely mirroring that of `NumPy`. This makes it a natural solution for Python developers who want to leverage the massive computational power of GPUs.

The main challenge in calculating the Jaro-Winkler similarity metric and, more generally, in handling strings on GPUs stems from the fact that the latter do not natively support jagged arrays, also called "arrays of arrays." A string is an array of characters. Thus, an array of strings is, in effect, an array of arrays of characters. This limitation similarly applies to "arrays of arrays" for other data types. A simple solution is to convert jagged arrays into a different data structure: the Arrow columnar format [@arrow_format]. Numerous libraries have adopted this format, including `PyArrow` and `RAPIDS cuDF` [@rapids]. In short, this approach consists of storing jagged arrays in a primitive layout, that is, a long array of contiguous values of the same data type and fixed memory size (e.g., a long array of characters), along with a sequence of indices that indicate the starting position of each inner array within the outer array. Concretely, with this approach, arrays of strings are flattened into a single array of characters. The character array and its index buffers can be efficiently stored and manipulated on GPUs. For example, the array of strings `['David', 'Elizabeth', 'James', 'Jennifer', 'John', 'Linda', 'Mary', 'Michael', 'Patricia', 'Robert']` can be represented as an array of characters `['D', 'a', 'v', 'i', 'd', 'E', 'l', 'i', 'z', 'a', 'b', 'e', 't', 'h', 'J', 'a', 'm', 'e', 's', 'J', 'e', 'n', 'n', 'i', 'f', 'e', 'r', 'J', 'o', 'h', 'n', 'L', 'i', 'n', 'd', 'a', 'M', 'a', 'r', 'y', 'M', 'i', 'c', 'h', 'a', 'e', 'l', 'P', 'a', 't', 'r', 'i', 'c', 'i', 'a', 'R', 'o', 'b', 'e', 'r', 't']`, along with the sequence of indices, `[0, 5, 14, 19, 27, 31, 36, 40, 47, 55]`. This strategy is efficient in terms of access patterns and memory usage.

![Performance Comparison Between `Fast-ER` and `fastLink` for Record Linkage \label{linkage}](Performance Comparison Record Linkage.svg)

To illustrate the performance of GPU-accelerated record linkage, we compare the performance of our library with that of the leading CPU-powered software implementation, `fastLink` [@fastLink; @ENAMORADO_FIFIELD_IMAI_2019]. We join two extracts of North Carolina voter registration rolls of varying sizes (from 1,000 to 100,000 observations), comparing first names, last names, house numbers, and street names for fuzzy matching and birth years for exact matching. The datasets have 50% overlapping records. We injected noise into 5% of the records through various transformations: character addition, character deletion, random shuffling of values, replacing a character with another, and swapping two adjacent characters. The results in \autoref{linkage} confirm that our GPU-accelerated implementation delivers speeds over 35 times faster than `fastLink`.

![Performance Comparison Between `Fast-ER` and `fastLink` for Deduplication \label{deduplication}](Performance Comparison Deduplication.svg)

Analogously, we compare the performance of our library for deduplication with that of the leading CPU-powered software implementation. Deduplication was executed on one of the datasets described above. The results in \autoref{deduplication} confirm that our GPU-accelerated implementation runs over 60 times faster than `fastLink`.

# References
