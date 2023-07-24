#!/usr/bin/env python

import os

from setuptools import find_packages, setup

version_py = os.path.join(os.path.dirname(__file__), "torchcfm", "__version__.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
setup(
    name="torchcfm",
    version=version,
    description="Conditional Flow Matching for Fast Continuous Normalizing Flow Training.",
    author="Alexander Tong",
    author_email="alexandertongdev@gmail.com",
    url="https://github.com/atong01/conditional-flow-matching",
    install_requires=["torch", "pot", "numpy", "torchdyn"],
    packages=find_packages(),
)
