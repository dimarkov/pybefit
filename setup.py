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
 'pybefit.inference.pyro',
 'pybefit.inference.numpyro',
 'pybefit.tasks']

package_data = \
{'': ['*']}

install_requires = \
['numpyro',
 'optax',
 'pyro-ppl']

extras_require = \
{
 'CM': ['pycm'],
 'vis': ['seaborn', 'matplotlib'],
 'jupyter': ['jupyterlab', 'jupyterthemes', 'jupyter-black']
}

classifiers = \
[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.10'
]

setup_kwargs = {
    'name': 'pybefit',
    'version': '0.1.23',
    'description': 'Probabilistic inference for models of behaviour',
    'long_description': 'PyBefit is a Python library for Bayesian analysis of behavioral data. It is based on Pyro/Numpyro a probabilistic programing language, PyTorch, and Jax machine learning libraries.',
    'author': 'Dimitrije Marković',
    'author_email': 'dimitrije.markovic@tu-dresden.de',
    'url': 'https://github.com/dimarkov/pybefit',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.10',
}

setup(**setup_kwargs)

