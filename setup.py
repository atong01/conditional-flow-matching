#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="torchcfm",
    version="1.0.0",
    description="Conditional Flow Matching for Fast Continuous Normalizing Flow Training.",
    author="Alexander Tong",
    author_email="alexandertongdev@gmail.com",
    url="https://github.com/atong01/conditional-flow-matching",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["torch", "pot", "numpy", "torchdyn"],
    packages=find_packages(),
)
