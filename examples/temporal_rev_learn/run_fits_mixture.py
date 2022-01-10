#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer paramaters and posterior probability of different models $m=(\nu_{min}, \nu_{max})$,
from behavioural data under the hierarchical mixture model.
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
from jax import random, nn, devices, device_put


# import utility functions for model inversion and belief estimation
from utils import estimate_beliefs, mixture_model, mixture_model_dynpref

# import data loader
from stats import load_data

outcomes_data, responses_data, mask_data, ns, _, _ = load_data()

def main(seed, device):
    m_data = jnp.array(mask_data)
    r_data = jnp.array(responses_data).astype(jnp.int32)
    o_data = jnp.array(outcomes_data).astype(jnp.int32)

    rng_key = random.PRNGKey(seed)
    cutoff_up = 800
    cutoff_down = 0

    priors = []
    params = []
    for m in range(1, 11):
        if m <= 10:
            seq, _ = estimate_beliefs(o_data, r_data, mask=m_data, nu_max=m)
        else:
            seq, _ = estimate_beliefs(o_data, r_data, mask=m_data, nu_max=10, nu_min=m-10)
        priors.append(seq['beliefs'][0][cutoff_down:cutoff_up])
        params.append(seq['beliefs'][1][cutoff_down:cutoff_up])

    c0 = jnp.sum(nn.one_hot(outcomes_data[:cutoff_down], 4) * jnp.expand_dims(mask_data[:cutoff_down], -1), 0)
    
    nuts_kernel = NUTS(mixture_model_dynpref)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000, num_chains=1, progress_bar=True)
    seqs = device_put(
                        (
                            jnp.stack(priors, 0), 
                            jnp.stack(params, 0), 
                            o_data[cutoff_down:cutoff_up], 
                            c0
                        ), 
                    device)

    mcmc.run(
        rng_key, 
        seqs, 
        y=device_put(r_data[cutoff_down:cutoff_up], device), 
        mask=device_put(m_data[cutoff_down:cutoff_up].astype(bool), device), 
        extra_fields=('potential_energy',)
    )
    
    samples = mcmc.get_samples()    

    jnp.savez('fit_waic_sample/dynpref_fit_sample_mixture.npz', samples=samples)

    print(mcmc.get_extra_fields()['potential_energy'].mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimates behavioral data")
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    npyro.set_platform(args.device)
    device = devices(args.device)[0]
    main(args.seed, device)
