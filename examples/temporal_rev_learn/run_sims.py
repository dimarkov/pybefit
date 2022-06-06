#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate behaviour for different parameter setup and perform 
model inversion based on simulated responses. Saves posterior
probabilities for different $\nu_{max}$.
@author: Dimitrije Markovic
"""
# set environment for better memory menagment
import os 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import numpy as np
import jax.numpy as jnp

import numpyro as npyro

from pybefit.agents import HSMMAI as Agent
from utils import estimate_beliefs, simulator, mixture_model
from jax import random, devices, device_put, nn, jit, vmap
from numpyro.infer import MCMC, NUTS
from scipy import io

def get_belief_sequence(outcomes, responses, m_inf, cutoff):
    # here we map model number m_inf to the parameter value pairs nu_max, nu_min

    def true_fn(m):
        seq, _ = estimate_beliefs(outcomes, responses, 'cpu', nu_max=m, nu_min=0)
        return seq

    def false_fn(m):
        seq, _ = estimate_beliefs(outcomes, responses, 'cpu', nu_max=10, nu_min=m-10)
        return seq

    if m_inf < 11:
        seq_sim = true_fn(m_inf)
    else:
        seq_sim = false_fn(m_inf)
    
    return (seq_sim['beliefs'][0][cutoff:], seq_sim['beliefs'][1][cutoff:])


def inference(belief_sequences, obs, mask, rng_key):
    nuts_kernel = NUTS(mixture_model, dense_mass=True)
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=200, 
        num_samples=200, 
        num_chains=5,
        chain_method="vectorized",
        progress_bar=False
    )

    mcmc.run(
        rng_key, 
        belief_sequences, 
        obs, 
        mask
    )

    samples = mcmc.get_samples()

    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison insilico data")
    parser.add_argument("-n", "--subjects", default=1, type=int)
    parser.add_argument("-s", "--seed", default=111, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    npyro.set_platform(args.device)

    device = devices(args.device)[0]
    
    rng_key = random.PRNGKey(args.seed)
    cutoff = 400
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

    outcomes = device_put(jnp.concatenate([outcomes1, outcomes2], -1), devices('cpu')[0])

    N = 2 * n
    T = len(outcomes)
    subs = device_put(jnp.array(range(N)), devices('cpu')[0])
    process = lambda t, responses: outcomes[t, subs, responses]

    # prior preferences 
    p_p = .6
    p_m = .1
    p_c = 1 - p_p - p_m
    P_o = [p_m, p_p, p_c/2, p_c/2]
    U = jnp.log(jnp.array(P_o))

    # m_rng = range(1, 11)  # comparison based on nu_max
    m_rng = [1,] + list(range(10, 20))  # comparison based on nu_min

    post_smpl = {}
    for m_true in m_rng:
        print('true model: ', m_true)
        if m_true < 11:
            agent = Agent(T, N, nu_max=m_true, U=U)
        else:
            agent = Agent(T, N, nu_max=10, nu_min=m_true-10, U=U)

        gamma = 5. * jnp.ones((N, 1, 1))
        sequences = simulator(process, agent, gamma, seed=m_true)
        responses_sim = sequences['choices']
        outcomes_sim = sequences['outcomes']

        priors = []
        params = []
        
        for m_inf in m_rng:
            seq_sim = get_belief_sequence(outcomes_sim, responses_sim, m_inf, cutoff)
            priors.append(seq_sim[0].copy())
            params.append(seq_sim[1].copy())

        c0 = np.sum(nn.one_hot(outcomes_sim[cutoff:], 4).copy().astype(float), 0)

        # fit simulated data
        seqs = device_put(
                    (
                        np.stack(priors, 0), 
                        np.stack(params, 0), 
                        outcomes_sim[cutoff:].copy(), 
                        c0
                    ),
                    device
                )

        y = device_put(responses_sim[cutoff:].copy(), device)
        mask = jnp.ones_like(y).astype(bool)

        rng_keys = random.split(rng_key, N)

        samples = jit(vmap(inference, in_axes=((2, 2, 1, 0), 1, 1, 0)))(seqs, y, mask, rng_keys)

        post_smpl[m_true] = samples

        jnp.savez('fit_sims/tmp_sims_m{}.npz'.format(m_true), samples=samples)

        del seq_sim, samples, agent

    # save posterior estimates
    jnp.savez('fit_sims/sims_mcomp_numin_P-{}-{}-{}-{}.npz'.format(*P_o), samples=post_smpl)

    # delete tmp files
    for m_true in m_rng:
        os.remove('fit_sims/tmp_sims_m{}.npz'.format(m_true))