---
title: 'Fast-ER: GPU-Accelerated Record Linkage in Python'
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
date: 10 October 2024
bibliography: paper.bib
---

<!---Adding references--->

# Summary

Record linkage, also called "entity resolution," consists of matching observations from two datasets representing the same unit, even when consistent common identifiers are absent. This process typically involves computing string similarity metrics, such as the Jaro-Winkler metric, for all pairs of values between the datasets. The `Fast-ER` package accelerates these computations with graphical processing units (GPUs). It estimates the parameters of the Fellegi-Sunter model, a widely used probabilistic record linkage model, and performs the necessary data processing on CUDA-enabled GPUs. Our experiments demonstrate that this approach can increase processing speed by over 60 times, reducing processing time from hours to minutes, compared to the previous leading software implementation. This significantly improves the scalability of probabilistic record linkage and deduplication for large datasets.

# Statement of Need

Record linkage usually involves calculating string similarity metrics for all possible pairs of values between two datasets. Although these calculations are simple, they become computationally expensive as the number of observations increases, causing the number of required comparisons to grow exponentially. For instance, when linking observations from two datasets, each with 1,000,000 observations, adding just one more observation to either dataset results in an additional 1,000,000 comparisons. This makes record linkage prohibitively expensive to perform, even for datasets of moderate size.

GPUs were developed in the 1970s to accelerate digital image processing. Unlike central processing units (CPUs), which excel at executing sequential instructions quickly, GPUs are optimized for performing hundreds of operations in parallel. Their inherent parallelism makes them particularly well-suited for tasks that can be divided into many simultaneous calculations. Early GPU applications focused on geometric transformations, such as rotating and translating vertices between coordinate systems, and texture mapping, where they could handle large numbers of pixels concurrently.

GPUs are highly effective not only for graphics but also for non-graphical computations. They excel at high-throughput computations involving data parallelism, where the same operations are applied to many data points at once. This stems from their Single Instruction, Multiple Data (SIMD) architecture, which allows them to act as stream or vector processors. By harnessing the computational power of modern shader pipelines, GPUs can execute "compute kernels," similar to the instructions in a "for loop," but instead of running sequentially, these operations are executed simultaneously across data points.  As a result, GPUs often deliver performance that is orders of magnitude faster than traditional CPUs, particularly for large-scale vector or matrix calculations.

Our GPU-accelerated implementation of probabilistic record linkage makes extensive use of the CuPy library `[@cupy_learningsys2017]`. CuPy is an open-source library designed for GPU-accelerated, array-based numerical computations in Python. Built on CUDA, NVIDIA's parallel computing platform and programming model, CuPy provides an intuitive API that closely resembles NumPy’s. This makes it an ideal choice for Python developers accustomed to the latter who want to leverage the computational power of GPUs.

The main challenge in implementing the Jaro-Winkler similarity metric, and more broadly in handling strings, on GPUs arises from the fact that they do not natively support jagged arrays, commonly referred to as "arrays of arrays." Since a string is effectively an array of characters, an array of strings represents an array of arrays of characters. This limitation extends to "arrays of arrays" for other data types.

To overcome this limitation, a simple solution is to convert jagged arrays into a different data structure: the Arrow columnar format. It has already been adopted by numerous libraries, including PyArrow and RAPIDS cuDF. In short, this approach involves storing "arrays of arrays" in a primitive layout, consisting of a long array of contiguous values of the same data type and fixed memory size (e.g., a long array of characters), paired with a sequence of buffers that mark the starting position of each inner array within the jagged array. For an array of strings, the arrays of characters that form the strings are flattened into a single array of characters. Both arrays can be stored and manipulated by GPUs. For example, the array of strings `['David', 'Elizabeth', 'James', 'Jennifer', 'John', 'Linda', 'Mary', 'Michael', 'Patricia', 'Robert']` can be represented as an array of characters `['D', 'a', 'v', 'i', 'd', 'E', 'l', 'i', 'z', 'a', 'b', 'e', 't', 'h', 'J', 'a', 'm', 'e', 's', 'J', 'e', 'n', 'n', 'i', 'f', 'e', 'r', 'J', 'o', 'h', 'n', 'L', 'i', 'n', 'd', 'a', 'M', 'a', 'r', 'y', 'M', 'i', 'c', 'h', 'a', 'e', 'l', 'P', 'a', 't', 'r', 'i', 'c', 'i', 'a', 'R', 'o', 'b', 'e', 'r', 't']`, along with the following sequence of buffers, `[0, 5, 14, 19, 27, 31, 36, 40, 47, 55]`. This strategy is efficient both in terms of memory usage and access patterns. 

<!---Add figures--->

To demonstrate the benefits of GPU-accelerated probabilistic record linkage, we compare the performance of our library with that of the previous leading software implementation, fastLink. We join two excerpts of North Carolina voter registration rolls of varying sizes (from 1,000 to 100,000 observations) along four variables: first name, last name, house number, and street name. Datasets have 50% overlapping records. To introduce the need for probabilistic record linkage, we inject noise into 5% of the records through various transformations: character addition, character deletion, random shuffling of values, replacing a character with another, and swapping two adjacent characters. The results confirm that our GPU-accelerated implementation of the Fellegi-Sunter probabilistic record linkage model is consistently faster than fastLink, delivering speed improvements exceeding 60 times.
