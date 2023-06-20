# Probabilistic inference for models of behaviour
=================================================

PyBefit is a Python library for Bayesian analysis of behavioral data. It is based on [Pyro/Numpyro](pyro.ai) a probabilistic programing language, [PyTorch](https://pytorch.org/), and [Jax](https://github.com/google/jax) machine learning libraries.

Requirements
------------
    pyro
    pytorch
    numpyro
    jax
    [Optional]
     - matplolib
     - seaborn
     - arviz
     - jupyterlab
    
Installation
------------
To install PyBefit with CPU-only versions of JAX and PyTorch you can run
```sh
pip install pip --upgrade
pip install pybefit --upgrade
```
### Conda installation
To install either JAX or Pytorch with Nvidia GPU support we recomment using [anaconda](https://conda.io/miniconda.html) package manager. First create pybefit environment and activate it
```sh
conda create -n befit python=3.11
conda activate befit
```
Then follow instructions for installing [JAX](https://github.com/google/jax#installation) or [Pytorch](https://pytorch.org/) with GPU support.

Finally install pybefit via pip within the conda environment
```sh
pip install pip --upgrade
pip install pybefit --upgrade
```

For development you can install pybefit directly from repo as follows
```sh
git clone https://github.com/dimarkov/pybefit.git
cd pybefit
pip install pip --upgrade
pip install -e .
```

Examples
--------
PyBefit is used as a basis for several projects:
 * The code and notebooks in `examples/control_dilemmas/` acompanies the following paper - [Marković et al. "Meta-control of the exploration-exploitation dilemma emerges from probabilistic inference over a hierarchy of time scales." Cognitive, Affective, & Behavioral Neuroscience (2020): 1-25](https://link.springer.com/article/10.3758/s13415-020-00837-x) and can be used to reproduce all the figures.

 * The code and notebooks in `examples/social_influence/` shows the analysis of behavioural data of subjects in different age groups performing the [social influence task](https://academic.oup.com/scan/article/12/4/618/2948767?login=true). We use a range of computational models of behaviour in changing environments and present their age group dependent comparison.

 * The code and notebooks in `examples/temporal_rev_learn` acompanies the following paper - [Marković, Dimitrije, Andrea MF Reiter, and Stefan J. Kiebel. "Revealing human sensitivity to a latent temporal structure of changes." (2022)](https://www.frontiersin.org/articles/10.3389/fnbeh.2022.962494). In the paper we analyse the behavioural data of subjects performing a temporally structured reversal learning task. The goal here is to demonstrate subjects' learning of latent temporal strucure of a noisy and dynamic environment.

 * The code and notebooks in `examples/plandepth` illustrates the model for inferring participants planning depth, while they are performing the Space Adventure Task [SAT](https://github.com/dimarkov/sat).

License
-------
See [license](LICENSE.md)

