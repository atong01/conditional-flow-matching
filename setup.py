#!/usr/bin/env python

import os

from setuptools import find_packages, setup

install_requires = [
    "torch>=1.11.0",
    "matplotlib",
    "numpy",  # Due to pandas incompatibility
    "scipy",
    "scikit-learn",
    "torchdyn>=1.0.6",
    "pot",
    "torchdiffeq",
    "absl-py",
    "pandas>=2.2.2",
]

version_py = os.path.join(os.path.dirname(__file__), "torchcfm", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
readme = open("README.md", encoding="utf8").read()
setup(
    name="torchcfm",
    version=version,
    description="Conditional Flow Matching for Fast Continuous Normalizing Flow Training.",
    author="Alexander Tong, Kilian Fatras",
    author_email="alexandertongdev@gmail.com",
    url="https://github.com/atong01/conditional-flow-matching",
    install_requires=install_requires,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    extras_require={"forest-flow": ["xgboost", "scikit-learn", "ForestDiffusion"]},
)
