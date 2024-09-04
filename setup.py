from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fast-er",
    version="0.1.0",
    description="GPU-Accelerated Probabilistic Record Linkage in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    install_requires=["cupy", "numpy", "pandas", "scikit-learn", "tensorly"],
)
