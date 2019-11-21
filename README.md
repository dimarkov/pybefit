# Probabilistic inference for models of behaviour
=================================================

PyBefit is a Python library for Bayesian analysis of behavioral data.
It is based on [Pyro](pyro.ai) a probabilistic programing language 
and [PyTorch](https://pytorch.org/) a machine learning library.

Requirements
------------

numpy
pandas
pytorch
matplotlib
seaborn

Installation
------------

The easiest way to install required libraries is using [conda](https://conda.io/miniconda.html)
and pip package managers.

First setup an environment using anaconda prompt (or just terminal in linux):

```sh
conda create -n befit python=3 numpy pandas matplotlib seaborn
conda activate befit
conda install pytorch -c pytorch
pip install pyro-ppl
```

License
-------

See [license](LICENSE.md)

