# Fast-ER: GPU-Accelerated Record Linkage in Python

[![Documentation Status](https://readthedocs.org/projects/fast-er/badge/?version=latest)](https://fast-er.readthedocs.io/en/latest/?badge=latest)

**Authors:**
- [Jacob Morrier](https://www.jacobmorrier.com)
- [Sulekha Kishore](https://www.linkedin.com/in/sulekha-kishore/)
- [R. Michael Alvarez](https://www.rmichaelalvarez.com)

Record linkage, also called "entity resolution," consists of identifying matching observations across different datasets, even when consistent common identifiers are missing. This process typically requires computing string similarity metrics, such as the Jaro-Winkler metric, for all pairs of values between the datasets. 

The Fast-ER package harnesses the computational power of graphical processing units (GPUs) to accelerate this process dramatically. It estimates the parameters of the widely used Fellegi-Sunter model and performs the necessary data preprocessing, including the computation of string similarity metrics, on CUDA-enabled GPUs. 

Fast-ER executes over 60 times faster than the previous leading software implementation, reducing processing time from hours to minutes. This significantly enhances the scalability of record linkage and deduplication for large datasets.
