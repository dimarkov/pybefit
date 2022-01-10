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

import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpyro as npyro
from numpyro.infer import MCMC, NUTS
from jax import random, nn

# import utility functions for model inversion and belief estimation
from utils import estimate_beliefs, dynamic_preference_gm

# import data loader
from stats import load_data

outcomes_data, responses_data, mask_data, ns, corrects, conditions = load_data()

mask_data = jnp.array(mask_data)
responses_data = jnp.array(responses_data).astype(jnp.int32)
outcomes_data = jnp.array(outcomes_data).astype(jnp.int32)

from scipy import io
states_and_rewards = io.loadmat('main/states_and_rewards.mat')
Sirr = states_and_rewards['irregular']['S'][0, 0][:, 0] - 1
Sreg = states_and_rewards['regular']['S'][0, 0][:, 0] - 1

ns_reg = ns[1]
ns_irr = ns[2]

states = []
for cond in conditions['condition']:
    if cond == 'regular':
        states.append( Sreg )
    else:
        states.append( Sirr )

states =  jnp.stack(states, 0)

def main(model, subject, seed):
    rng_key = random.PRNGKey(seed)
    cutoff_up = 800
    cutoff_down = 0

    rspns = responses_data[..., subject:subject+1]
    outcms = outcomes_data[..., subject:subject+1]
    msk = mask_data[..., subject:subject+1]

    if model <= 10:
        seq, _ = estimate_beliefs(outcms, rspns, mask=msk, nu_max=model)
    else:
        seq, _ = estimate_beliefs(outcms, rspns, mask=msk, nu_max=10, nu_min=model-10)
    
    gm = dynamic_preference_gm

    # init preferences
    c0 = jnp.sum(nn.one_hot(outcms[:cutoff_down], 4) * jnp.expand_dims(msk[:cutoff_down], -1), 0)

    nuts_kernel = NUTS(gm)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000)
    
    seq = (seq['beliefs'][0][cutoff_down:cutoff_up], 
           seq['beliefs'][1][cutoff_down:cutoff_up], 
           outcms[cutoff_down:cutoff_up],
           c0)

    rng_key, _rng_key = random.split(rng_key)
    mcmc.run(
        _rng_key, 
        seq, 
        y=rspns[cutoff_down:cutoff_up], 
        mask=msk[cutoff_down:cutoff_up].astype(bool), 
        extra_fields=('potential_energy',)
    )
    
    samples = mcmc.get_samples()

    true_state = states[subject][cutoff_down:cutoff_up]
    res_probs = nn.softmax(samples['logits'], -1)
    sim_res = random.categorical(rng_key, jnp.log(res_probs)).squeeze(-1)
    sim_corr = sim_res == true_state
    sim_corr = jnp.where(sim_res == 2, jnp.nan, sim_corr)

    print('subject {} in condition {}'.format(subject, conditions['condition'].loc[subject]))
    
    true_state = nn.one_hot(jnp.abs(true_state-1), 2)
    corr_prob = jnp.sum(res_probs[..., 0, :-1] * true_state, -1)
    print(mcmc.get_extra_fields()['potential_energy'].mean())
    return (samples, 
            seq, 
            rspns[cutoff_down:cutoff_up], 
            corr_prob, 
            corrects[cutoff_down:cutoff_up, subject], 
            sim_corr)

from stats import performance
if __name__ == "__main__":
    npyro.set_platform('gpu')

    model = 1
    seed = 2

    for subject in range(26, 27):    
        samples, beliefs, responses, corr_probs, corr, sim_corr = main(model, subject, seed)

        print(responses[..., 0][jnp.isnan(corr)])

        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        axes[0].plot(performance(corr[:, None], ws=201).squeeze(), lw=3)
        perf = performance(sim_corr.T, ws=201).T
        axes[0].plot(perf, 'r', alpha=0.05)
        axes[0].set_ylabel('performance')
        axes[1].plot(samples['gamma'].squeeze().T, 'r', alpha=0.01)
        axes[1].set_ylabel('gamma')
        axes[1].set_xlabel('trial')
        plt.show()

