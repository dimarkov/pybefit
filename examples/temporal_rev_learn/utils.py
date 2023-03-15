#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for simulations and inference
@author: Dimitrije Markovic
"""

import jax.numpy as jnp
import numpyro as npyro
import numpyro.distributions as dist
from jax import random, lax, nn
from numpyro.infer import log_likelihood
from numpyro.distributions import TransformedDistribution, transforms

from opt_einsum import contract
from numpyro.contrib.control_flow import scan

from pybefit.agents.jax.hsmm_ai import HSMMAI as  Agent 
from pybefit.agents.jax.hsmm_ai import logits

def einsum(equation, *args):
    return contract(equation, *args, backend='jax')

def simulator(process, agent, gamma, seed=0, **model_kw):
    # POMDP simulator
    def sim_fn(carry, t):
        rng_key, prior = carry
        
        rng_key, _rng_key = random.split(rng_key)

        choices = agent.action_selection(_rng_key, prior, gamma=gamma[..., t, :], **model_kw)
        
        outcomes = process(t, choices)
        
        posterior = agent.learning(t, outcomes, choices, prior)
                
        return (rng_key, posterior), {'outcomes': outcomes, 
                                      'choices': choices,
                                      'logits': agent.logits}
    
    rng_key = random.PRNGKey(seed)
    _, sequence = lax.scan(sim_fn, (rng_key, agent.prior), jnp.arange(agent.T))
    sequence['outcomes'].block_until_ready()
    
    return sequence

def estimate_beliefs(outcomes, choices, device, mask=1, nu_max=10, nu_min=0, **kwargs):
    # belief estimator from fixed responses and outcomes
    T, N = choices.shape
    assert outcomes.shape == (T, N)
    agent = Agent(T, N, nu_max=nu_max, nu_min=nu_min, mask=mask, device=device, prior_kwargs=kwargs)
    def sim_fn(carry, t):
        prior = carry
        posterior = agent.learning(t, outcomes[t], choices[t], prior)

        p_cfm, params = prior

        p_c = einsum('...cfm->...c', p_cfm)
                
        return posterior, {'beliefs': (p_c, params)}
    
    _, sequence = lax.scan(sim_fn, agent.prior, jnp.arange(T))
    
    sequence['beliefs'][0].block_until_ready()
    
    return sequence, agent

def generative_model(beliefs, y=None, mask=True):
    # generative model
    T, N = beliefs[0].shape[:2]
    with npyro.plate('N', N):
        gamma = npyro.sample('gamma', dist.Gamma(20., 2.))

        td = TransformedDistribution(
            dist.Normal(jnp.array([-1., .7]), jnp.array([1., 0.2])).to_event(1), 
            transforms.OrderedTransform()
        )

        lams = npyro.sample('lams', td)
        U = jnp.pad(lams, ((0, 0), (0, 2)), 'constant', constant_values=(0,))
        
        with npyro.plate('T', T):
            logs = npyro.deterministic('logits', logits(beliefs, jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2)) )
            npyro.sample('y', dist.CategoricalLogits(logs).mask(mask), obs=y)

def log_pred_density(model, samples, *args, **kwargs):
    # waic score of posterior samples
    log_lk = log_likelihood(model, samples, *args, **kwargs)['y']
    ll = log_lk.sum(-1)

    S = ll.shape[0]
    lppd = nn.logsumexp(ll, 0) - jnp.log(S)
    p_waic = jnp.var(ll, axis=0, ddof=1)
    return lppd - p_waic, log_lk

def mixture_model(beliefs, y, mask, weights):
    M, T, N, _ = beliefs[0].shape

    assert weights.shape == (N, M)

    gamma = npyro.sample('gamma', dist.InverseGamma(2., 2.).expand([N]))
    
    p = npyro.sample('p', dist.Dirichlet(jnp.ones(3)).expand([N]))
    
    p0 = npyro.deterministic('p0', jnp.concatenate([p[..., :1]/2, p[..., :1]/2 + p[..., 1:2], p[..., 2:]/2, p[..., 2:]/2], -1))

    U = jnp.log(p0)

    def transition_fn(carry, t):
        lgts = logits((beliefs[0][:, t], beliefs[1][:, t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        mixing_dist = dist.CategoricalProbs(weights)
        component_dist = dist.CategoricalLogits(lgts.swapaxes(1, 0)).mask(mask[t, :, None])
        with npyro.plate('subject', N):
            npyro.sample('y', dist.MixtureSameFamily(mixing_dist, component_dist))

        return None, None

    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, None, jnp.arange(T)
        )

def aux_mixture_model(beliefs, y, mask, weights):
    M, T, N, _ = beliefs[0].shape

    assert weights.shape == (N, M)

    gamma = npyro.sample('gamma', dist.InverseGamma(2., 2.).expand([N]))
    
    p = npyro.sample('p', dist.Dirichlet(jnp.ones(3)).expand([N]))
    
    p0 = npyro.deterministic(
        'p0', 
        jnp.concatenate(
            [p[..., :1]/2, 
            p[..., :1]/2 + p[..., 1:2], 
            p[..., 2:]/2, 
            p[..., 2:]/2], 
            -1
        )
    )

    U = jnp.log(p0)

    lgts = logits(
        (beliefs[0], beliefs[1]),
        jnp.expand_dims(gamma, -1),
        jnp.expand_dims(U, -2)
    )

    cat_dist = dist.CategoricalLogits(lgts).mask(mask)

    with npyro.plate('subject', N):
        with npyro.plate('step', T):
            with npyro.plate('model', M):
                npyro.sample('y', cat_dist, obs=y)

def complete_mixture_model(beliefs, y, mask, condition, aux=False):
    M, T, N, _ = beliefs[0].shape
    
    tau = npyro.sample('tau', dist.HalfCauchy(1.))
    r_1 = npyro.sample('r_1', dist.Dirichlet(jnp.ones(M)/tau))
    r_2 = npyro.sample('r_2', dist.Dirichlet(jnp.ones(M)/tau))
    
    weights = jnp.stack([r_1, r_2], 0)[condition]
    if aux:
        aux_mixture_model(beliefs, y, mask, weights)
    else:
        mixture_model(beliefs, y, mask, weights)