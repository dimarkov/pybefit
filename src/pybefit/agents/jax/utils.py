#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for defining agents
@author: Dimitrije Markovic
"""

import jax.numpy as jnp

from opt_einsum import contract

def einsum(equation, *args):
    return contract(equation, *args, backend='jax')

def log(x, minimum=1e-45):
    return jnp.log(jnp.clip(x, minimum))