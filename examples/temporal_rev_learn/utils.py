#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for simulations and inference
@author: Dimitrije Markovic
"""

import jax.numpy as jnp
import numpyro as npyro
import numpyro.distributions as dist
from jax import random, lax
from numpyro.infer import log_likelihood
from jax.scipy.special import logsumexp
from agents import Agent, logits

def simulator(process, agent, seed=0, **model_kw):
    # POMDP simulator
    def sim_fn(carry, t):
        rng_key, prior = carry
        
        rng_key, _rng_key = random.split(rng_key)
        choices = agent.action_selection(_rng_key, prior, **model_kw)
        
        outcomes = process(t, choices)
        
        posterior = agent.learning(t, outcomes, choices, prior)
                
        return (rng_key, posterior), {'outcomes': outcomes, 
                                      'choices': choices}
    
    rng_key = random.PRNGKey(seed)
    _, sequence = lax.scan(sim_fn, (rng_key, agent.prior), jnp.arange(agent.T))
    sequence['outcomes'].block_until_ready()
    
    return sequence

def estimate_beliefs(outcomes, choices, mask=1, nu_max=10, nu_min=0):
    # belief estimator from fixed responses and outcomes
    T, N = choices.shape
    assert outcomes.shape == (T, N)
    agent = Agent(T, N, nu_max=nu_max, nu_min=nu_min, mask=mask)
    def sim_fn(carry, t):
        prior = carry
        posterior = agent.learning(t, outcomes[t], choices[t], prior)
                
        return posterior, {'beliefs': prior}
    
    _, sequence = lax.scan(sim_fn, agent.prior, jnp.arange(T))
    
    sequence['beliefs'][0].block_until_ready()
    
    return sequence, agent

def generative_model(beliefs, y=None, mask=True):
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

        with npyro.plate('T', T):
            logs = logits(beliefs, jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))
            npyro.sample('obs', dist.CategoricalLogits(logs).mask(mask), obs=y)

def log_pred_density(model, samples, *args, **kwargs):
    # waic score of posterior samples
    log_lk = log_likelihood(model, samples, *args, **kwargs)['obs'].sum(-2)
    S = log_lk.shape[0]
    lppd = logsumexp(log_lk, 0) - jnp.log(S)
    p_waic = jnp.var(log_lk, axis=0, ddof=1)
    return {'lpd': lppd, 'waic': lppd - p_waic}


from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import validate_sample

class CategoricalMixture(Distribution):
    arg_constraints = {'logits': constraints.real_vector}
    has_enumerate_support = False
    is_discrete = True

    def __init__(self, logits, weights, mask, validate_args=None):
        if jnp.ndim(logits) < 1:
            raise ValueError("`logits` parameter must be at least one-dimensional.")
        self.logits = logits
        self.weights = weights
        self.mask = mask
        super(CategoricalMixture, self).__init__(batch_shape=jnp.shape(weights)[:-1],
                                                validate_args=validate_args)
    def sample(self, key, sample_shape=()):
        raise NotImplementedError 

    @validate_sample
    def log_prob(self, value):
        mask = jnp.expand_dims(self.mask, -1)
        one_hot_value = jnp.eye(3)[value]
        logits = jnp.sum(jnp.expand_dims(one_hot_value, -1) * self.logits, -2) * mask
        log_z = logsumexp(self.logits, -2) * mask
        log_pmf = logsumexp(jnp.sum(logits - log_z, -3) + jnp.log(self.weights), -1)

        return log_pmf

    @property
    def support(self):
        return constraints.integer_interval(0, jnp.shape(self.logits)[-2] - 1)

def mixture_model(sequences, agents, mask, y=None):
    D = len(sequences)
    T, N = sequences[1]['beliefs'][0].shape[:2]
    
    tau = npyro.sample('tau', dist.HalfCauchy(1.))
    with npyro.plate('N', N):
        weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(D)))
        with npyro.plate('D', D):
            gamma = npyro.sample('gamma', dist.Gamma(20., 2.))
            prob = npyro.sample('prob', dist.Dirichlet(jnp.array([1., 5., 4.])))

        def vec_func(i, nu):
            U = jnp.log(jnp.stack([prob[i, :, 0], prob[i, :, 1], prob[i, :, 2]/2, prob[i, :, 2]/2], -1))
            res = agents[nu].logits(
                    sequences[nu]['beliefs'], 
                    jnp.expand_dims(gamma[i], -1), 
                    1., 
                    jnp.expand_dims(U, -2)
                  )
            return res
        logits = []
        for i, nu in enumerate(sequences):
            logits.append(vec_func(i, nu))

        obs = npyro.sample('obs', CategoricalMixture(logits, weights, mask), obs=y)