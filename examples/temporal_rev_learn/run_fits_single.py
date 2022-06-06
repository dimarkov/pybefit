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
import numpyro as npyro

def main(m, seed, device, dynamic_gamma, dynamic_preference):
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from jax import random, nn, vmap, jit, devices, device_put

    # import utility functions for model inversion and belief estimation
    from utils import estimate_beliefs, single_model, log_pred_density

    # import data loader
    from stats import load_data

    outcomes_data, responses_data, mask_data, ns, _, _ = load_data()

    mask_data = jnp.array(mask_data)
    responses_data = jnp.array(responses_data).astype(jnp.int32)
    outcomes_data = jnp.array(outcomes_data).astype(jnp.int32)

    rng_key = random.PRNGKey(seed)
    cutoff_up = 1000
    cutoff_down = 400

    print(m, seed, dynamic_gamma, dynamic_preference)
    
    if m <= 10:
        seq, _ = estimate_beliefs(outcomes_data, responses_data, device, mask=mask_data, nu_max=m)
    else:
        seq, _ = estimate_beliefs(outcomes_data, responses_data, device, mask=mask_data, nu_max=10, nu_min=m-10)
    
    model = lambda *args: single_model(*args, dynamic_gamma=dynamic_gamma, dynamic_preference=dynamic_preference)

    # init preferences
    c0 = jnp.sum(nn.one_hot(outcomes_data[:cutoff_down], 4) * jnp.expand_dims(mask_data[:cutoff_down], -1), 0)

    if dynamic_gamma:
        num_warmup = 500
        num_samples = 500
        num_chains = 2
    else:
        num_warmup = 100
        num_samples = 100
        num_chains = 10

    def inference(belief_sequences, obs, mask, rng_key):
        nuts_kernel = NUTS(model, dense_mass=True)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method="vectorized", progress_bar=False)

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

    seq = (
        seq['beliefs'][0][cutoff_down:cutoff_up], 
        seq['beliefs'][1][cutoff_down:cutoff_up], 
        outcomes_data[cutoff_down:cutoff_up],
        c0
    )

    y = responses_data[cutoff_down:cutoff_up]
    mask = mask_data[cutoff_down:cutoff_up].astype(bool)

    n = mask.shape[-1]
    rng_keys = random.split(rng_key, n)
    samples, potential_energy = jit(vmap(inference, in_axes=((1, 1, 1, 0), 1, 1, 0)))(seq, y, mask, rng_keys)

    waic = vmap(lambda *args: log_pred_density(model, *args), in_axes=(0, (1, 1, 1, 0), 1, 1))

    waic, log_likelihood = waic(samples, seq, y, mask)

    print('waic', waic.mean())

    jnp.savez('fit_data/fit_waic_sample_minf{}_gamma{}_pref{}_long.npz'.format(m, int(dynamic_gamma), int(dynamic_preference)), samples=samples, waic=waic, log_likelihood=log_likelihood)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimates behavioral data")
    parser.add_argument("-m", "--model", default=1, type=int)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("-dg", "--dynamic-gamma", action='store_true', help="make gamma parameter dynamic")
    parser.add_argument("-dp", "--dynamic-preference", action='store_true', help="make preference parameters dynamic")
    args = parser.parse_args()

    npyro.set_platform(args.device)
    npyro.enable_x64()
    main(args.model, args.seed, args.device, args.dynamic_gamma, args.dynamic_preference)