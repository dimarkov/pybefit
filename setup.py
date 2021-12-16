# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['pybefit', 'pybefit.agents', 'pybefit.inference', 'pybefit.tasks']

package_data = \
{'': ['*']}

install_requires = \
['jupyter[black]>=1.0.0,<2.0.0',
 'matplotlib>=3.5.1,<4.0.0',
 'nb-black>=1.0.7,<2.0.0',
 'numpy>=1.21.4,<2.0.0',
 'numpyro>=0.8.0,<0.9.0',
 'pandas>=1.3.5,<2.0.0',
 'pyro-ppl>=1.8.0,<2.0.0',
 'seaborn>=0.11.2,<0.12.0']

setup_kwargs = {
    'name': 'pybefit',
    'version': '0.1.0',
    'description': 'PyBefit is a Python library for Bayesian analysis of behavioral data.',
    'long_description': None,
    'author': 'Dimitrije Markovic',
    'author_email': 'dimitrije.markovic@tu-dresden.de',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.11',
}


setup(**setup_kwargs)

