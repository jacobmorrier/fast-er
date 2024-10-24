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
date: 24 October 2024
bibliography: paper.bib
---

# Summary

Record linkage, also called "entity resolution," consists of identifying matching observations across different datasets, even when consistent common identifiers are missing. On the other hand, deduplication consists of identifying duplicate entries within a dataset when consistent common identifiers are missing. These processes typically involve computing string similarity metrics, such as the Jaro-Winkler metric, for all pairs of values between the datasets.

The `Fast-ER` package harnesses the computational power of graphical processing units (GPUs) to accelerate these processes dramatically. It estimates the parameters of the widely used Fellegi-Sunter model and performs the necessary data preprocessing, especially the calculation of string similarity metrics, on CUDA-enabled GPUs.

`Fast-ER` executes over 35 times faster than the previously leading software implementation, reducing processing time from hours to minutes. This significantly enhances the scalability of record linkage and deduplication for large datasets.

# Statement of Need

Record linkage and deduplication typically involve calculating string similarity metrics, such as the Jaro-Winkler metric [@Winkler_1990], for all pairs of values between two datasets. Although these calculations are simple, the number of comparisons grows exponentially with the number of observations. For instance, when linking observations from two datasets, each with 1,000,000 observations, adding just one more observation to either dataset results in an additional 1,000,000 comparisons. This makes record linkage and deduplication prohibitively expensive to perform, even for datasets of moderate size.

GPUs were first developed in the 1970s to accelerate digital image processing. Unlike central processing units (CPUs), designed for the sequential execution of a single thread of instructions with minimal latency, GPUs are optimized for performing hundreds of operations simultaneously [@kirk_hwu_2017]. Early applications focused on geometric transformations, such as rotating and translating vertices between coordinate systems, and texture mapping. GPUs can also be used for non-graphical computations. They are especially well-suited for high-throughput computations that can be broken down into identical, independent calculations, such as those exhibiting data parallelism, where the same instructions are applied individually to many data points. This stems from GPUs’ Single Instruction, Multiple Data (SIMD) architecture. Concretely, the shader pipelines of modern GPUs can execute "compute kernels," analogous to instructions in a "for loop." However, rather than running sequentially, these operations are executed simultaneously across inputs. As a result, GPUs can often deliver performance orders of magnitude faster than traditional CPUs.

Our GPU-accelerated implementation of record linkage makes extensive use of the `CuPy` library [@cupy_learningsys2017]. This open-source library is designed for GPU-accelerated, array-based numerical computations in Python. Built on NVIDIA's `CUDA` parallel computing platform, `CuPy` has an intuitive application programming interface (API) that closely mirrors that of `NumPy`. This similarity makes it a natural fit for Python developers who want to leverage the computational power of GPUs.

The primary challenge in calculating the Jaro-Winkler similarity metric, and more broadly in handling strings, on GPUs stems from the fact that they do not natively support jagged arrays, often called "arrays of arrays." Since a string is effectively an array of characters, an array of strings represents an array of arrays of characters. This limitation extends to "arrays of arrays" for other data types. To overcome this limitation, a simple solution is to convert jagged arrays into a different data structure: the Arrow columnar format [@arrow_format]. Numerous libraries have adopted this format, including `PyArrow` and `RAPIDS cuDF` [@rapids]. In short, this approach involves storing jagged arrays in a primitive layout, consisting of a long array of contiguous values of the same data type and fixed memory size (e.g., a long array of characters), paired with a sequence of buffers that mark the starting position of each inner array within the outer array. For an array of strings, the arrays of characters that form the strings are flattened into a single array of characters. Both arrays can be stored and manipulated by GPUs. For example, the array of strings `['David', 'Elizabeth', 'James', 'Jennifer', 'John', 'Linda', 'Mary', 'Michael', 'Patricia', 'Robert']` can be represented as an array of characters `['D', 'a', 'v', 'i', 'd', 'E', 'l', 'i', 'z', 'a', 'b', 'e', 't', 'h', 'J', 'a', 'm', 'e', 's', 'J', 'e', 'n', 'n', 'i', 'f', 'e', 'r', 'J', 'o', 'h', 'n', 'L', 'i', 'n', 'd', 'a', 'M', 'a', 'r', 'y', 'M', 'i', 'c', 'h', 'a', 'e', 'l', 'P', 'a', 't', 'r', 'i', 'c', 'i', 'a', 'R', 'o', 'b', 'e', 'r', 't']`, along with the following sequence of buffers, `[0, 5, 14, 19, 27, 31, 36, 40, 47, 55]`. This strategy is efficient both in terms of access patterns and memory usage.

![Performance Comparison Between `Fast-ER` and `fastLink` for Record Linkage](Performance Comparison Record Linkage.svg)

![Performance Comparison Between `Fast-ER` and `fastLink` for Deduplication](Performance Comparison Deduplication.svg)

To illustrate the benefits of GPU-accelerated record linkage, we compare the performance of our library with that of the previously leading software implementation, `fastLink` [@fastLink; @ENAMORADO_FIFIELD_IMAI_2019]. We join two excerpts of North Carolina voter registration rolls of varying sizes (from 1,000 to 100,000 observations), comparing first names, last names, house numbers, and street names for fuzzy matching and birth years for exact matching. The datasets have 50% overlapping records. We injected noise into 5% of the records through various transformations: character addition, character deletion, random shuffling of values, replacing a character with another, and swapping two adjacent characters. The results confirm that our GPU-accelerated implementation is consistently faster than `fastLink`, delivering speed improvements exceeding 35 times.

# References
