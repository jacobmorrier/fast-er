from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="fast-er-link",
    version="0.2.0",
    description="GPU-Accelerated Record Linkage and Deduplication in Python",
    long_description="Fast-ER is a Python package for GPU-accelerated record linkage and deduplication.",
    url="https://github.com/jacobmorrier/fast-er",
    author="Jacob Morrier, Sulekha Kishore, R. Michael Alvarez",
    author_email="jmorrier@caltech.edu, sulekha@caltech.edu, rma@hss.caltech.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    keywords="entity resolution, GPU, probabilistic record linkage, record linkage",
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["cupy-cuda12x", "numpy", "pandas", "matplotlib"],
)
