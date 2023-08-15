"""Setup file for package installation."""

from setuptools import setup, find_packages

requirements = [
    "setuptools >= 61.0.0",
    # "pennylane >= 0.32.0-dev0",
    "numpy < 1.24",  # leave until pennylane 0.32.0 is released
    "jax",
    "jaxlib",
    "optax",
    "networkx",
    "PyYAML",
    "matplotlib",
    "tqdm",
    "scipy",
]

info = {
    "name": "QT-CorGI",
    "version": "0.1.0",
    "url": "https://github.com/QSAR-UBC/qutrit-qaoa-dev",
    "license": "MIT license",
    "author": "Gabriel Bottrill",
    "author_email": "bottrill@student.ubc.ca",
    "packages": find_packages(where="."),
    "description": "Comparing qubits and qutrits for solving 3-colouring using QAOA",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "provides": ["qtcorgi"],
    "install_requires": requirements,
}

classifiers = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
