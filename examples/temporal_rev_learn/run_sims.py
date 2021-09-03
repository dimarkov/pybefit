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
import numpyro.distributions as dist

from agents import Agent
from utils import estimate_beliefs, simulator, log_pred_density
from utils import generative_model as model
from jax import random
from numpyro.infer import MCMC, NUTS
from scipy import io

import os 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

def get_belief_sequence(outcomes, responses, generator, m_inf, cutoff):
    if m_inf < 11:
        seq_sim, _ = estimate_beliefs(outcomes, 
                                      responses, 
                                      nu_max=m_inf, 
                                      nu_min=0)
    else:
        seq_sim, _ = estimate_beliefs(outcomes, 
                                      responses, 
                                      nu_max=10, 
                                      nu_min=m_inf-10)
    
    return (seq_sim['beliefs'][0][-cutoff:], seq_sim['beliefs'][1][-cutoff:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison insilico data")
    parser.add_argument("-n", "--subjects", default=1, type=int)
    parser.add_argument("-m", "--model", default=1, type=int)  # models are enumerated from 1 to 19
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    npyro.set_platform(args.device)
    
    rng_key = random.PRNGKey(args.seed)
    cutoff = 800
    m_inf = args.model
    n = args.subjects

    # load experiment and generate observations
    data = io.loadmat('main/states_and_rewards.mat')
    Sirr = data['irregular']['S'][0, 0][:, 0] - 1
    Oirr = data['irregular']['R'][0, 0]
    Sreg = data['regular']['S'][0, 0][:, 0] - 1
    Oreg = data['regular']['R'][0, 0]

    outcomes1 = jnp.concatenate([(Oreg[:, None].repeat(n, -2) + 1)//2, 
                                 (Oirr[:, None].repeat(n, -2) + 1)//2], -2)
    outcomes2 = jnp.concatenate([Sreg[:, None].repeat(n, -1) + 2, 
                                 Sirr[:, None].repeat(n, -1) + 2], -1)[..., None]

    outcomes = jnp.concatenate([outcomes1, outcomes2], -1)

    N = 2 * n
    T = len(outcomes)
    subs = jnp.array(range(N))
    process = lambda t, responses: outcomes[t, subs, responses]

    # prior preferences 
    U = jnp.array([-.5, 1.5, 0., 0.])

    res_waic = {m_inf: {}}
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000, progress_bar=False)
    for m_true in range(1, 20):
        print('true model: ', m_true, 'fitted model: ', m_inf)
        if m_true < 11:
            agent = Agent(T, N, nu_max=m_true, U=U)
        else:
            agent = Agent(T, N, nu_max=10, nu_min=m_true-10, U=U)

        sequences = simulator(process, agent, gamma=10.)
        responses_sim = sequences['choices']
        outcomes_sim = sequences['outcomes']
        rng_key, _rng_key = random.split(rng_key)
        seq_sim = get_belief_sequence(outcomes_sim, responses_sim, m_inf, cutoff)

        # fit simulated data
        mcmc.run(_rng_key, seq_sim, y=responses_sim[-cutoff:])
        sample = mcmc.get_samples()

        # estimate WAIC/posterior predictive log likelihood
        res_waic[m_inf][m_true] = log_pred_density(model, sample, seq_sim, y=responses_sim[-cutoff:])['waic']
        del seq_sim, sample

    # save waic scores
    jnp.savez('waic_sim_minf_{}.npz'.format(m_inf), waic=res_waic)