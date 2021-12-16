# Probabilistic inference for models of behaviour
=================================================

PyBefit is a Python library for Bayesian analysis of behavioral data. It is based on [Pyro/Numpyro](pyro.ai) a probabilistic programing language and [PyTorch](https://pytorch.org/) and [Jax](https://github.com/google/jax) machine learning libraries.

Requirements
------------

numpy
pandas
pyro
pytorch
numpyro
jax
jaxlib
matplotlib
seaborn

Installation
------------
The easiest way to install required libraries and the PyBefit package is using [poetry](https://conda.io/miniconda.html) package manager. Inside the project root directory run 
```sh
poetry install
poetry shell
```

If cuda support is present and required for your work simply run the following commands
```sh
poetry shell poe force-cuda11-torch
poetry shell poe force-cuda11-jax
```
This will upgrade pytorch and jax to the version with cuda 
support. 

Examples
--------

License
-------

See [license](LICENSE.md)

