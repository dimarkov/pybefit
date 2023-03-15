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

from pybefit.agents.jax.hsmm_ai import HSMMAI as Agent
from utils import estimate_beliefs, simulator, complete_mixture_model
from jax import lax, random, devices, device_put, nn
from numpyro.infer import MCMC, NUTS, log_likelihood
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


def inference(belief_sequences, obs, mask, conditions, rng_key):
    nuts_kernel = NUTS(complete_mixture_model, dense_mass=True)
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=200, 
        num_samples=100, 
        num_chains=10,
        chain_method="vectorized",
        progress_bar=True
    )

    mcmc.run(
        rng_key, 
        belief_sequences, 
        obs, 
        mask,
        conditions
    )

    samples = mcmc.get_samples()

    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison insilico data")
    parser.add_argument("-n", "--subjects", default=1, type=int)
    parser.add_argument("-s", "--seed", default=111, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("-i", default=0, type=int)  # select which device to use for simulations and inference
    args = parser.parse_args()

    npyro.set_platform(args.device)

    device = devices(args.device)[args.i]
    
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

    m_max = 10
    m_rng = range(1, m_max + 1)  # comparison based on nu_max
    # m_rng = range(1, 20)  # comparison based on nu_max and nu_min

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
        
        for m_inf in range(1, m_max + 1):
            seq_sim = get_belief_sequence(outcomes_sim, responses_sim, m_inf, cutoff)
            priors.append(seq_sim[0].copy())
            params.append(seq_sim[1].copy())

        c0 = np.sum(nn.one_hot(outcomes_sim[cutoff:], 4).copy().astype(float), 0)

        # fit simulated data
        seqs = device_put(
                    (
                        np.stack(priors, 0), 
                        np.stack(params, 0), 
                        outcomes_sim[cutoff:], 
                        c0
                    ),
                    device
                )
        seqs = lax.stop_gradient(seqs)
        y = lax.stop_gradient(device_put(responses_sim[cutoff:], device))
        mask = lax.stop_gradient(device_put(jnp.ones_like(y).astype(bool), device))

        conditions = lax.stop_gradient(
            device_put(jnp.concatenate([jnp.zeros(n, dtype=jnp.int16), jnp.ones(n, dtype=jnp.int16)]), device)
        )
        rng_key, _rng_key = device_put(random.split(rng_key), device)

        samples = lax.stop_gradient(inference(seqs, y, mask, conditions, _rng_key))
        for key in samples:
            print(key, samples[key].shape)

        samples = device_put(samples, devices('cpu')[0])
        seqs = device_put(seqs, devices('cpu')[0])
        y = device_put(y, devices('cpu')[0])
        mask = device_put(mask, devices('cpu')[0])
        conditions = device_put(conditions, devices('cpu')[0])

        log_ll = log_likelihood(
            complete_mixture_model, samples, seqs, y, mask, conditions, parallel=True, aux=True
        )['y'].sum(-2)

        r = jnp.stack([samples['r_1'], samples['r_2']], -1)[..., conditions]

        log_prod = log_ll + jnp.log(r)
        post_marg = nn.softmax(log_prod, -2)
        exceedance_prob = nn.one_hot(jnp.argmax(post_marg, -2), num_classes=m_max).mean(0)

        print('tau', samples['tau'].mean(), samples['tau'].std())
        print('rs', samples['r_1'].mean(0), samples['r_2'].mean(0))
        print('ep', exceedance_prob[:n].mean(-2), exceedance_prob[n:].mean(-2))

        samples['EP'] = exceedance_prob

        post_smpl[m_true] = samples

        jnp.savez('fit_sims/tmp_sims_m{}.npz'.format(m_true), samples=samples)

        del seq_sim, samples, agent

    # save posterior estimates
    jnp.savez('results/fit_sims/sims_mcomp_P-{}-{}-{}-{}.npz'.format(*P_o), samples=post_smpl)

    # delete tmp files
    for m_true in m_rng:
        os.remove('results/fit_sims/tmp_sims_m{}.npz'.format(m_true))