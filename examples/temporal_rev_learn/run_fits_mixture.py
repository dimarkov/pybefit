#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer paramaters and posterior probability of different models $M=(\nu_{min}, \nu_{max})$,
from behavioural data under the hierarchical mixture model.
@author: Dimitrije Markovic
"""
# set environment for better memory menagment
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import argparse
import numpyro as npyro

def main(seed, device, dynamic_gamma, dynamic_preference, mc_type):

    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from jax import random, nn, devices, device_put, vmap, jit

    # import utility functions for model inversion and belief estimation
    from utils import estimate_beliefs, mixture_model

    # import data loader
    from stats import load_data

    outcomes_data, responses_data, mask_data, ns, _, _ = load_data()

    print(seed, device, dynamic_gamma, dynamic_preference)

    model = lambda *args: mixture_model(*args, dynamic_gamma=dynamic_gamma, dynamic_preference=dynamic_preference)

    m_data = jnp.array(mask_data)
    r_data = jnp.array(responses_data).astype(jnp.int32)
    o_data = jnp.array(outcomes_data).astype(jnp.int32)

    rng_key = random.PRNGKey(seed)
    cutoff_up = 1000
    cutoff_down = 400

    priors = []
    params = []

    if mc_type == 'nu_max':
        M_rng = list(range(1, 11))  # model comparison for regular condition
    else:
        M_rng = [1,] + list(range(11, 20))  # model comparison for irregular condition

    for M in M_rng:
        if M <= 10:
            seq, _ = estimate_beliefs(o_data, r_data, device, mask=m_data, nu_max=M)
        else:
            seq, _ = estimate_beliefs(o_data, r_data, device, mask=m_data, nu_max=10, nu_min=M-10)
        
        priors.append(seq['beliefs'][0][cutoff_down:cutoff_up])
        params.append(seq['beliefs'][1][cutoff_down:cutoff_up])
    
    device = devices(device)[0]

    # init preferences
    c0 = jnp.sum(nn.one_hot(outcomes_data[:cutoff_down], 4) * jnp.expand_dims(mask_data[:cutoff_down], -1), 0)

    if dynamic_gamma:
        num_warmup = 1000
        num_samples = 1000
        num_chains = 1
    else:
        num_warmup = 200
        num_samples = 200
        num_chains = 5

    def inference(belief_sequences, obs, mask, rng_key):
        nuts_kernel = NUTS(model, dense_mass=True)
        mcmc = MCMC(
            nuts_kernel, 
            num_warmup=num_warmup, 
            num_samples=num_samples, 
            num_chains=num_chains,
            chain_method="vectorized",
            progress_bar=False
        )

        mcmc.run(
            rng_key, 
            belief_sequences, 
            obs, 
            mask, 
            extra_fields=('potential_energy',)
        )

        samples = mcmc.get_samples()
        potential_energy = mcmc.get_extra_fields()['potential_energy'].mean()
        # mcmc.print_summary()

        return samples, potential_energy

    seqs = device_put(
                    (
                        jnp.stack(priors, 0), 
                        jnp.stack(params, 0), 
                        o_data[cutoff_down:cutoff_up], 
                        c0
                    ), 
                device)

    y = device_put(r_data[cutoff_down:cutoff_up], device)
    mask = device_put(m_data[cutoff_down:cutoff_up].astype(bool), device)

    n = mask.shape[-1]
    rng_keys = random.split(rng_key, n)

    samples, potential_energy = jit(vmap(inference, in_axes=((2, 2, 1, 0), 1, 1, 0)))(seqs, y, mask, rng_keys)

    print('potential_energy', potential_energy)     

    jnp.savez('fit_data/fit_sample_mixture_gamma{}_pref{}_{}.npz'.format(int(dynamic_gamma), int(dynamic_preference), mc_type), samples=samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimates behavioral data")
    parser.add_argument("-s", "--seed", default=110, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("-dg", "--dynamic-gamma", action='store_true', help="make gamma parameter dynamic")
    parser.add_argument("-dp", "--dynamic-preference", action='store_true', help="make preference parameters dynamic")
    parser.add_argument("-mc", "--mc-type", default="nu_max", type=str, help='use "nu_max" or "nu_min".')
    args = parser.parse_args()

    npyro.set_platform(args.device)
    npyro.enable_x64()
    main(args.seed, args.device, args.dynamic_gamma, args.dynamic_preference, args.mc_type)
