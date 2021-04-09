#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate behaviour for different parameter setup and perform 
model inversion based on simulated responses. Saves waic scores 
for different models.
@author: Dimitrije Markovic
"""
import argparse
import gc
import jax.numpy as jnp

import numpyro as npyro
import numpyro.distributions as dist

from agents import Agent, logits
from jax import lax, random
from numpyro.infer import MCMC, NUTS
from numpyro.infer import log_likelihood
from jax.scipy.special import logsumexp
from scipy import io

def simulator(process, agent, seed=0, **model_kw):
    # POMDP simulator
    def sim_fn(carry, t):
        rng_key, prior = carry
        
        rng_key, _rng_key = random.split(rng_key)
        choices = agent.action_selection(_rng_key, prior, **model_kw)
        
        outcomes = process(t, choices)
        
        posterior = agent.learning(outcomes, choices, prior)
                
        return (rng_key, posterior), {'outcomes': outcomes, 
                                      'choices': choices}
    
    rng_key = random.PRNGKey(seed)
    _, sequence = lax.scan(sim_fn, (rng_key, agent.prior), jnp.arange(len(outcomes)))
    sequence['outcomes'].block_until_ready()
    
    return sequence

def estimate_beliefs(outcomes, choices, nu_max=10, nu_min=0):
    # belief estimator from fixed responses and outcomes
    T, N = choices.shape
    assert outcomes.shape == (T, N)
    agent = Agent(N, nu_max=nu_max, nu_min=nu_min)
    def sim_fn(carry, t):
        prior = carry
        posterior = agent.learning(outcomes[t], choices[t], prior)
                
        return posterior, {'beliefs': prior}
    
    _, sequence = lax.scan(sim_fn, agent.prior, jnp.arange(T))
    
    sequence['beliefs'][0].block_until_ready()
    
    return sequence, agent

def model(beliefs, y=None, mask=True):
    # generative model
    T, N = beliefs[0].shape[:2]
    with npyro.plate('N', N):
        gamma = npyro.sample('gamma', dist.Gamma(20., 2.))
        lams = npyro.sample('lams', dist.Normal(jnp.array([-1., 1.]), 
                                                2*jnp.ones(2)).to_event(1))
        U = jnp.stack([lams[..., 0], 
                       lams[..., 1], 
                       jnp.zeros_like(lams[..., 1]), 
                       jnp.zeros_like(lams[..., 1])], -1)
        U -= logsumexp(U)
        with npyro.plate('T', T):
            logs = logits(beliefs, jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))
            obs = npyro.sample('obs', dist.CategoricalLogits(logs).mask(mask), obs=y)

def log_pred_density(model, samples, *args, **kwargs):
    # waic score of posterior samples
    log_lk = log_likelihood(model, samples, *args, **kwargs)['obs'].sum(-2)
    n = log_lk.shape[0]
    _lpd = logsumexp(log_lk, 0) - jnp.log(n)
    p_waic = n * log_lk.var(0) / (n - 1)
    return {'lpd': _lpd, 'waic': _lpd - p_waic}

def get_data_and_agent(outcomes, responses, generator, mixed, nu, cutoff):
    if generator == 'max' and mixed:
        seq_sim, agent_sim = estimate_beliefs(outcomes, 
                                              responses, 
                                              nu_max=10, 
                                              nu_min=nu-1)
    elif generator == 'min' and mixed:
        seq_sim, agent_sim = estimate_beliefs(outcomes, 
                                              responses, 
                                              nu_max=nu, 
                                              nu_min=0)
    elif generator == 'max':
        seq_sim, agent_sim = estimate_beliefs(outcomes, 
                                              responses, 
                                              nu_max=nu, 
                                              nu_min=0)
    elif generator == 'min':
        seq_sim, agent_sim = estimate_beliefs(outcomes, 
                                              responses, 
                                              nu_max=10, 
                                              nu_min=nu-1)
    
    return (seq_sim['beliefs'][0][-cutoff:], seq_sim['beliefs'][1][-cutoff:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison insilico data")
    parser.add_argument("-n", "--subjects", default=1, type=int)
    parser.add_argument("-nu", "--precision", default=1, type=int)
    parser.add_argument("-g", "--generator", default='max', type=str)
    parser.add_argument("-m", "--mixing", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    npyro.set_platform(args.device)
    
    rng_key = random.PRNGKey(23478875)
    cutoff = 800
    nu_inf = args.precision
    n = args.subjects
    generator = args.generator
    mixed = args.mixing
    print(n, generator, mixed)
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
    subs = jnp.array(range(N))
    def process(t, responses):
        return outcomes[t, subs, responses]

    res_waic = {nu_inf: {}}
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000, progress_bar=False)
    for nu_true in range(1, 11):
        print(nu_true, nu_inf)
        if generator=='max':
            agent = Agent(N, nu_max=nu_true)
        else:
            agent = Agent(N, nu_max=10, nu_min=nu_true-1)

        sequences = simulator(process, agent, U=jnp.array([-.5, 1.5, 0., 0.]))
        responses_sim = sequences['choices']
        outcomes_sim = sequences['outcomes']
        rng_key, _rng_key = random.split(rng_key)
        seq_sim = get_data_and_agent(outcomes_sim, responses_sim, generator, mixed, nu_inf, cutoff)

        # fit simulated data
        mcmc.run(_rng_key, seq_sim, y=responses_sim[-cutoff:])
        sample = mcmc.get_samples()

        # estimate WAIC/posterior predictive log likelihood
        res_waic[nu_inf][nu_true] = log_pred_density(model, sample, seq_sim, y=responses_sim[-cutoff:])['waic']
        del seq_sim, sample
        gc.collect()

    # save waic scores U=jnp.array([-.5, 1.5, 0., 0.])
    jnp.savez('waic_sim_{}_{}_nu_{}.npz'.format(generator, mixed, nu_inf), waic=res_waic)

    # save waic scores U=jnp.array([1.5, 3.5, 0., 0.])
    # jnp.savez('waic_sim_{}_{}.npz'.format(generator, mixed), waic=res_waic)
