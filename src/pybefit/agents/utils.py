#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module contains various utility functions.

@author: Dimitrije Markovic
"""

__all__ = [
    'transform'
]

def transform(z, model, **kwargs):
    """Map unconstrained parameter values 'z' to constrained model parameters
    """
    runs, num_params = z.shape[-2:]
    # map z variables to model parameters
    agent = model(runs=runs, **kwargs)
    assert num_params == agent.num_params
    agent.set_parameters(z)

    return agent