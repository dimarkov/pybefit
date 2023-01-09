# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['pybefit',
 'pybefit.agents',
 'pybefit.agents.jax',
 'pybefit.agents.torch',
 'pybefit.inference',
 'pybefit.tasks']

package_data = \
{'': ['*']}

install_requires = \
['jax',
 'numpy',
 'numpyro',
 'optax',
 'torch',
 'pyro-ppl',
 'pandas',
 'pycm',
 'seaborn',
 'matplotlib',
 'arviz']

classifiers = \
[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10'
]

setup_kwargs = {
    'name': 'pybefit',
    'version': '0.1.0',
    'description': 'Probabilistic inference for models of behaviour',
    'long_description': 'PyBefit is a Python library for Bayesian analysis of behavioral data. It is based on Pyro/Numpyro a probabilistic programing language, PyTorch, and Jax machine learning libraries.',
    'author': 'Dimitrije Markovic',
    'author_email': 'dimitrije.markovic@tu-dresden.de',
    'url': 'https://github.com/dimarkov/pybefit',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.11',
}

setup(**setup_kwargs)

