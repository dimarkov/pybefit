# Probabilistic inference for models of behaviour
=================================================

PyBefit is a Python library for Bayesian analysis of behavioral data. It is based on [Pyro/Numpyro](pyro.ai) a probabilistic programing language, [PyTorch](https://pytorch.org/), and [Jax](https://github.com/google/jax) machine learning libraries.

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

If cuda support is present and required for your work, simply run the following commands
```sh
poetry shell pip install pip --upgrade
poetry shell poe force-cuda11-torch
poetry shell poe force-cuda11-jax
```
This will upgrade pytorch and jax to the version with cuda 
support.

If you prefer different environment managers you can install pybefit using the provided 'setup.py' with 
```sh
python setup.py develop
```

Examples
--------
PyBefit is used as a basis for several projects:
 * The code and notebooks in `examples/control_dilemmas/` acompanies the following paper - [Marković et al. "Meta-control of the exploration-exploitation dilemma emerges from probabilistic inference over a hierarchy of time scales." Cognitive, Affective, & Behavioral Neuroscience (2020): 1-25](https://link.springer.com/article/10.3758/s13415-020-00837-x) and can be used to reproduce all the figures.

 * The code and notebooks in `examples/social_influence/` shows the analysis of behavioural data of subjects in different age groups performing the [social influence task](https://academic.oup.com/scan/article/12/4/618/2948767?login=true). We use a range of computational models of behaviour in changing environments and present their age group dependent comparison.

 * The code and notebooks in `examples/temporal_rev_learn` shows analysis of unpublished behavioural data of subjects 
 performing a [temporally structured reversal learning task](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006707). The goal here is to demonstrate particpants learning of latent temporal strucure of a noisy dynamic environment using a model-based analysis.

License
-------
See [license](LICENSE.md)

