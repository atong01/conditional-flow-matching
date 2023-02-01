#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Conditional Flow Matching for Fast Continuous Normalizing Flow Training.",
    author="Alexander Tong",
    author_email="alexandertongdev@gmail.com",
    url="https://github.com/atong01/conditional-flow-matching",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
