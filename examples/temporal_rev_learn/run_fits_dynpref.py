#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer paramaters and estimate waic scores from behavioural data.
@author: Dimitrije Markovic
"""
# set environment for better memory menagment
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import jax.numpy as jnp

import numpyro as npyro
from numpyro.infer import MCMC, NUTS
from jax import random, nn

# import utility functions for model inversion and belief estimation
from utils import estimate_beliefs, dynamic_generative_model2, log_pred_density

from numpyro.infer.reparam import TransformReparam
from numpyro.infer import reparam 

# import data loader
from stats import load_data

outcomes_data, responses_data, mask_data, ns = load_data()

mask_data = jnp.array(mask_data)
responses_data = jnp.array(responses_data).astype(jnp.int32)
outcomes_data = jnp.array(outcomes_data).astype(jnp.int32)

def main(m, seed):
    rng_key = random.PRNGKey(seed)
    cutoff_up = 800
    cutoff_down = 100
    
    if m <= 10:
        seq, _ = estimate_beliefs(outcomes_data, responses_data, mask=mask_data, nu_max=m)
    else:
        seq, _ = estimate_beliefs(outcomes_data, responses_data, mask=mask_data, nu_max=10, nu_min=m-10)
    
    model = dynamic_generative_model2

    # init preferences
    c0 = jnp.sum(nn.one_hot(outcomes_data[:cutoff_down], 4) * jnp.expand_dims(mask_data[:cutoff_down], -1), 0)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000)
    seq = (seq['beliefs'][0][cutoff_down:cutoff_up], 
           seq['beliefs'][1][cutoff_down:cutoff_up], 
           outcomes_data[cutoff_down:cutoff_up],
           c0)
    
    rng_key, _rng_key = random.split(rng_key)
    mcmc.run(
        _rng_key, 
        seq, 
        y=responses_data[cutoff_down:cutoff_up], 
        mask=mask_data[cutoff_down:cutoff_up].astype(bool), 
        extra_fields=('potential_energy',)
    )
    
    samples = mcmc.get_samples()    
    waic = log_pred_density(
        model, 
        samples, 
        seq, 
        y=responses_data[cutoff_down:cutoff_up], 
        mask=mask_data[cutoff_down:cutoff_up].astype(bool)
    )['waic']

    jnp.savez('fit_waic_sample/dynpref_fit_waic_sample_minf{}.npz'.format(m), samples=samples, waic=waic)

    print(mcmc.get_extra_fields()['potential_energy'].mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimates behavioral data")
    parser.add_argument("-m", "--model", default=1, type=int)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    npyro.set_platform(args.device)
    main(args.model, args.seed)
