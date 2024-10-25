# Fast-ER: GPU-Accelerated Record Linkage and Deduplication in Python

[![Documentation Status](https://readthedocs.org/projects/fast-er/badge/?version=latest)](https://fast-er.readthedocs.io/en/latest/?badge=latest)

**Authors:**
- [Jacob Morrier](https://www.jacobmorrier.com)
- [Sulekha Kishore](https://www.linkedin.com/in/sulekha-kishore/)
- [R. Michael Alvarez](https://www.rmichaelalvarez.com)

`Fast-ER` is a package for GPU-accelerated record linkage and deduplication in Python.

Record linkage, also called "entity resolution," consists of identifying matching records across different datasets, even when no consistent common identifiers are available. Deduplication, on the other hand, consists of identifying duplicate entries within a dataset when consistent unique identifiers are inconsistent or missing. Both tasks typically involve computing string similarity metrics, such as the Jaro-Winkler metric, for all pairs of values between the datasets.

The `Fast-ER` package harnesses the computational power of graphical processing units (GPUs) to dramatically accelerate these processes. It estimates the widely used Fellegi-Sunter model and performs the computationally intensive preprocessing steps, including the calculation of string similarity metrics, on CUDA-enabled GPUs.

`Fast-ER` runs over 35 times faster than the leading CPU-powered software implementation, reducing processing time from hours to minutes. This significantly enhances the scalability of record linkage and deduplication for large datasets.
