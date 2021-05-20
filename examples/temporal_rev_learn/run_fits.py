#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate behaviour for different parameter setup and perform 
model inversion based on simulated responses. Saves waic scores 
for different models.
@author: Dimitrije Markovic
"""
import argparse
import jax.numpy as jnp

import numpyro as npyro
from numpyro.infer import MCMC, NUTS
from jax import random
from time import time_ns


# import utility functions for model inversion and belief estimation
from utils import estimate_beliefs, generative_model, log_pred_density

# import data loader
from stats import load_data

outcomes_data, responses_data, mask_data, ns = load_data()

mask_data = jnp.array(mask_data)
responses_data = jnp.array(responses_data).astype(jnp.int32)
outcomes_data = jnp.array(outcomes_data).astype(jnp.int32)

def main(m):
    rng_key = random.PRNGKey(time_ns())
    cutoff = 800
    
    if m <= 10:
        seq, _ = estimate_beliefs(outcomes_data, responses_data, mask=mask_data, nu_max=m)
    else:
        seq, _ = estimate_beliefs(outcomes_data, responses_data, mask=mask_data, nu_max=10, nu_min=m-10)
    
    nuts_kernel = NUTS(generative_model)
    mcmc = MCMC(nuts_kernel, num_warmup=1500, num_samples=1000)
    seq = (seq['beliefs'][0][-cutoff:], 
           seq['beliefs'][1][-cutoff:])
    
    rng_key, _rng_key = random.split(rng_key)
    mcmc.run(
        _rng_key, 
        seq, 
        y=responses_data[-cutoff:], 
        mask=mask_data[-cutoff:].astype(bool), 
        extra_fields=('potential_energy',)
    )
    
    samples = mcmc.get_samples()    
    waic = log_pred_density(generative_model, samples, seq, y=responses_data[-cutoff:], mask=mask_data[-cutoff:].astype(bool))['waic']

    jnp.savez('fit_waic_sample_{}.npz'.format(m), samples=samples, waic=waic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimates behavioral data")
    parser.add_argument("-m", "--model", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    npyro.set_platform(args.device)
    main(args.model)
