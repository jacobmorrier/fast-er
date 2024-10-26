# Fast-ER: GPU-Accelerated Record Linkage and Deduplication in Python

[![Documentation Status](https://readthedocs.org/projects/fast-er/badge/?version=latest)](https://fast-er.readthedocs.io/en/latest/?badge=latest)

**Authors:**
- [Jacob Morrier](https://www.jacobmorrier.com)
- [Sulekha Kishore](https://www.linkedin.com/in/sulekha-kishore/)
- [R. Michael Alvarez](https://www.rmichaelalvarez.com)

`Fast-ER` is a Python package for GPU-accelerated record linkage and deduplication.

Record linkage, or "entity resolution," consists of identifying matching records across multiple datasets that lack common unique identifiers. Deduplication, on the other hand, involves recognizing duplicate entries within a dataset in which unique identifiers are either inconsistent or missing.

Both tasks often require calculating string similarity metrics, such as the Jaro-Winkler metric, for all pairs of values between datasets. The Fast-ER package harnesses the computational power of graphical processing units (GPUs) to accelerate this dramatically. It estimates the widely used Fellegi-Sunter probabilistic model and performs the computationally intensive preprocessing steps, including calculating string similarity metrics, on CUDA-enabled GPUs.

Fast-ER runs over 35 times faster than the leading CPU-powered software implementation, reducing execution time from hours to minutes. This significantly enhances the scalability of record linkage and deduplication for large datasets.
