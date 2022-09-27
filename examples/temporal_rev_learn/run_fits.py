#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer paramaters and posterior probability of different models $M=(\nu_{min}, \nu_{max})$,
from behavioural data under the hierarchical mixture model.
@author: Dimitrije Markovic
"""
# set environment for better memory menagment
import os
from syslog import LOG_PERROR
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import argparse
import numpyro as npyro

def main(seed, device, dynamic_gamma, dynamic_preference):

    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, log_likelihood
    from jax import lax, random, nn, devices, device_put

    # import utility functions for model inversion and belief estimation
    from utils import estimate_beliefs, complete_mixture_model

    # import data loader
    from stats import load_data

    outcomes_data, responses_data, mask_data, _, _, df = load_data()

    print(seed, device, dynamic_gamma, dynamic_preference)

    model = complete_mixture_model

    m_data = jnp.array(mask_data)
    r_data = jnp.array(responses_data).astype(jnp.int32)
    o_data = jnp.array(outcomes_data).astype(jnp.int32)

    rng_key = random.PRNGKey(seed)
    cutoff_up = 1000
    cutoff_down = 400

    priors = []
    params = []

    M_rng = list(range(1, 11))

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
        num_samples = 100
        num_chains = 10

    def inference(belief_sequences, obs, mask, conditions, rng_key):
        nuts_kernel = NUTS(model, dense_mass=True)
        mcmc = MCMC(
            nuts_kernel, 
            num_warmup=num_warmup, 
            num_samples=num_samples, 
            num_chains=num_chains,
            chain_method="vectorized",
            progress_bar=True
        )

        mcmc.run(
            rng_key, 
            belief_sequences, 
            obs, 
            mask,
            conditions, 
            extra_fields=('potential_energy',)
        )

        samples = mcmc.get_samples()
        potential_energy = mcmc.get_extra_fields()['potential_energy'].mean()
        # mcmc.print_summary()

        return samples, potential_energy

    seqs = lax.stop_gradient(
                device_put(
                    (
                        jnp.stack(priors, 0), 
                        jnp.stack(params, 0), 
                        o_data[cutoff_down:cutoff_up], 
                        c0
                    ), 
                    device
                )
            )

    y = lax.stop_gradient(device_put(r_data[cutoff_down:cutoff_up], device))
    mask = lax.stop_gradient(device_put(m_data[cutoff_down:cutoff_up].astype(bool), device))

    conditions = lax.stop_gradient(
            device_put( jnp.array((df.condition == 'irregular').values, dtype=jnp.int16), device)
        )

    samples, potential_energy = inference(seqs, y, mask, conditions, rng_key)

    print('potential_energy', potential_energy)
    
    # move arrays to cpu
    samples = device_put(samples, devices('cpu')[0])
    seqs = device_put(seqs, devices('cpu')[0])
    y = device_put(y, devices('cpu')[0])
    mask = device_put(mask, devices('cpu')[0])
    conditions = device_put(conditions, devices('cpu')[0])
    
    log_ll = log_likelihood(
        model, samples, seqs, y, mask, conditions, parallel=True, aux=True
    )['y'].sum(-2)

    r = jnp.stack([samples['r_1'], samples['r_2']], -1)[..., conditions]

    log_prod = log_ll + jnp.log(r)
    weights = nn.softmax(log_prod, -2)

    samples['weights'] = weights
    
    exceedance_prob = nn.one_hot(jnp.argmax(weights, -2), num_classes=len(M_rng)).mean(0)
    samples['EP'] = exceedance_prob

    jnp.savez('fit_data/fit_sample_gamma{}_pref{}.npz'.format(int(dynamic_gamma), int(dynamic_preference)), samples=samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model estimates behavioral data")
    parser.add_argument("-s", "--seed", default=110, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("-dg", "--dynamic-gamma", action='store_true', help="make gamma parameter dynamic")
    parser.add_argument("-dp", "--dynamic-preference", action='store_true', help="make preference parameters dynamic")
    args = parser.parse_args()

    npyro.set_platform(args.device)
    main(args.seed, args.device, args.dynamic_gamma, args.dynamic_preference)