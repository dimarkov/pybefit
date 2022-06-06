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

from pybefit.agents import HSMMAI as  Agent 
from pybefit.agents import logits

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

def single_model(beliefs, y, mask, dynamic_gamma=False, dynamic_preference=False):

    if dynamic_gamma: 
        if dynamic_preference:
            fulldyn_single_model(beliefs, y, mask)
        else:
            gammadyn_single_model(beliefs, y, mask)
    else:
        if dynamic_preference:
            prefdyn_single_model(beliefs, y, mask)
        else:
            nondyn_single_model(beliefs, y, mask)

def fulldyn_single_model(beliefs, y, mask):
    T, _ = beliefs[0].shape

    c0 = beliefs[-1]
    
    mu = npyro.sample('mu', dist.Normal(5., 5.))

    lam12 = npyro.sample('lam12', dist.HalfCauchy(1.).expand([2]).to_event(1))
    lam34 = npyro.sample('lam34', dist.HalfCauchy(1.))

    _lam34 = jnp.expand_dims(lam34, -1)
    lam0 = npyro.deterministic('lam0', jnp.concatenate([lam12.cumsum(-1), _lam34, _lam34], -1))

    eta = npyro.sample('eta', dist.Beta(1, 10))


    scale = npyro.sample('scale', dist.HalfNormal(1.))
    theta = npyro.sample('theta', dist.HalfCauchy(5.))
    rho = jnp.exp(- theta)
    sigma = jnp.sqrt( (1 - rho**2) / (2 * theta) ) * scale

    x0 = jnp.zeros(1)

    def transition_fn(carry, t):
        lam_prev, x_prev = carry

        gamma = npyro.deterministic('gamma', nn.softplus(mu + x_prev))

        U = jnp.log(lam_prev) - jnp.log(lam_prev.sum(-1, keepdims=True))

        logs = logits((beliefs[0][t], beliefs[1][t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        lam_next = npyro.deterministic('lams', lam_prev + nn.one_hot(beliefs[2][t], 4) * jnp.expand_dims(mask[t] * eta, -1))

        npyro.sample('y', dist.CategoricalLogits(logs).mask(mask[t]))
        noise = npyro.sample('dw', dist.Normal(0., 1.))

        x_next = rho * x_prev + sigma * noise

        return (lam_next, x_next), None

    lam_start = npyro.deterministic('lam_start', lam0 + jnp.expand_dims(eta, -1) * c0)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, (lam_start, x0), jnp.arange(T)
        )


def gammadyn_single_model(beliefs, y, mask):
    T, _ = beliefs[0].shape

    gamma = npyro.sample('gamma', dist.InverseGamma(2., 2.))
    mu = jnp.log(jnp.exp(gamma) - 1)


    p = npyro.sample('p', dist.Dirichlet(jnp.ones(3)))
    p0 = npyro.deterministic('p0', 
        jnp.concatenate([p[..., :1]/2, p[..., :1]/2 + p[..., 1:2], p[..., 2:]/2, p[..., 2:]/2], -1)
    )

    scale = npyro.sample('scale',  dist.Gamma(1., 1.))
    rho = npyro.sample('rho', dist.Beta(1., 2.))
    sigma = jnp.sqrt( - (1 - rho**2) / (2 * jnp.log(rho)) ) * scale

    U = jnp.log(p0)

    def transition_fn(carry, t):
        x_prev = carry

        dyn_gamma = npyro.deterministic('dyn_gamma', nn.softplus( mu + x_prev ))

        logs = logits(
            (beliefs[0][t], beliefs[1][t]),
            jnp.expand_dims(dyn_gamma, -1),
            jnp.expand_dims(U, -2)
        )

        npyro.sample('y', dist.CategoricalLogits(logs).mask(mask[t]))
        noise = npyro.sample('dw', dist.Normal(0., 1.))

        x_next = rho * x_prev + sigma * noise

        return x_next, None

    x0 = jnp.zeros(1)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, x0, jnp.arange(T)
        )


def prefdyn_single_model(beliefs, y, mask):
    T, _ = beliefs[0].shape

    c0 = beliefs[-1]
    
    lam12 = npyro.sample('lam12', dist.HalfCauchy(1.).expand([2]).to_event(1))
    lam34 = npyro.sample('lam34', dist.HalfCauchy(1.))

    _lam34 = jnp.expand_dims(lam34, -1)
    lam0 = npyro.deterministic('lam0', jnp.concatenate([lam12.cumsum(-1), _lam34, _lam34], -1))

    eta = npyro.sample('eta', dist.Beta(1, 10))

    gamma = npyro.sample('gamma', dist.InverseGamma(2., 2.))

    def transition_fn(carry, t):
        lam_prev = carry

        U = jnp.log(lam_prev) - jnp.log(lam_prev.sum(-1, keepdims=True))

        logs = logits((beliefs[0][t], beliefs[1][t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        lam_next = npyro.deterministic('lams', lam_prev + nn.one_hot(beliefs[2][t], 4) * jnp.expand_dims(mask[t] * eta, -1))

        npyro.sample('y', dist.CategoricalLogits(logs).mask(mask[t]))

        return lam_next, None

    lam_start = npyro.deterministic('lam_start', lam0 + jnp.expand_dims(eta, -1) * c0)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, lam_start, jnp.arange(T)
        )


def nondyn_single_model(beliefs, y, mask):
    T, _ = beliefs[0].shape

    gamma = npyro.sample('gamma', dist.InverseGamma(2., 3.))

    p = npyro.sample('p', dist.Dirichlet(jnp.ones(3)))
    p0 = npyro.deterministic('p0', 
        jnp.concatenate([p[..., :1]/2, p[..., :1]/2 + p[..., 1:2], p[..., 2:]/2, p[..., 2:]/2], -1)
    )

    U = jnp.log(p0)

    def transition_fn(carry, t):

        logs = logits(
            (beliefs[0][t], beliefs[1][t]), 
            jnp.expand_dims(gamma, -1), 
            jnp.expand_dims(U, -2)
        )
        npyro.sample('y', dist.CategoricalLogits(logs).mask(mask[t]))

        return None, None

    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, None, jnp.arange(T)
        )


def mixture_model(beliefs, y, mask, dynamic_gamma=False, dynamic_preference=False):

    if dynamic_gamma: 
        if dynamic_preference:
            fulldyn_mixture_model(beliefs, y, mask)
        else:
            gammadyn_mixture_model(beliefs, y, mask)
    else:
        if dynamic_preference:
            prefdyn_mixture_model(beliefs, y, mask)
        else:
            nondynamic_mixture_model(beliefs, y, mask)


def fulldyn_mixture_model(beliefs, y, mask):
    M, T, N, _ = beliefs[0].shape

    c0 = beliefs[-1]
    
    tau = .5
    with npyro.plate('N', N):
        weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(M)))

        assert weights.shape == (N, M)

        mu = npyro.sample('mu', dist.Normal(5., 5.))

        lam12 = npyro.sample('lam12', dist.HalfCauchy(1.).expand([2]).to_event(1))
        lam34 = npyro.sample('lam34', dist.HalfCauchy(1.))

        _lam34 = jnp.expand_dims(lam34, -1)
        lam0 = npyro.deterministic('lam0', jnp.concatenate([lam12.cumsum(-1), _lam34, _lam34], -1))

        eta = npyro.sample('eta', dist.Beta(1, 10))


        scale = npyro.sample('scale', dist.HalfNormal(1.))
        theta = npyro.sample('theta', dist.HalfCauchy(5.))
        rho = jnp.exp(- theta)
        sigma = jnp.sqrt( (1 - rho**2) / (2 * theta) ) * scale

    x0 = jnp.zeros(N)

    def transition_fn(carry, t):
        lam_prev, x_prev = carry

        gamma = npyro.deterministic('gamma', nn.softplus(mu + x_prev))

        U = jnp.log(lam_prev) - jnp.log(lam_prev.sum(-1, keepdims=True))

        logs = logits((beliefs[0][:, t], beliefs[1][:, t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        lam_next = npyro.deterministic('lams', lam_prev + nn.one_hot(beliefs[2][t], 4) * jnp.expand_dims(mask[t] * eta, -1))

        mixing_dist = dist.CategoricalProbs(weights)
        component_dist = dist.CategoricalLogits(logs.swapaxes(0, 1)).mask(mask[t][..., None])
        with npyro.plate('subjects', N):
            y = npyro.sample('y', dist.MixtureSameFamily(mixing_dist, component_dist))
            noise = npyro.sample('dw', dist.Normal(0., 1.))

        x_next = rho * x_prev + sigma * noise

        return (lam_next, x_next), None

    lam_start = npyro.deterministic('lam_start', lam0 + jnp.expand_dims(eta, -1) * c0)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, (lam_start, x0), jnp.arange(T)
        )

def prefdyn_mixture_model(beliefs, y, mask):
    M, T, N, _ = beliefs[0].shape

    c0 = beliefs[-1]
    
    tau = .5
    with npyro.plate('N', N):
        weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(M)))

        assert weights.shape == (N, M)

        mu = npyro.sample('mu', dist.Normal(5., 5.))

        lam12 = npyro.sample('lam12', dist.HalfCauchy(1.).expand([2]).to_event(1))
        lam34 = npyro.sample('lam34', dist.HalfCauchy(1.))

        _lam34 = jnp.expand_dims(lam34, -1)
        lam0 = npyro.deterministic('lam0', jnp.concatenate([lam12.cumsum(-1), _lam34, _lam34], -1))

        eta = npyro.sample('eta', dist.Beta(1, 10))
        gamma = npyro.deterministic('gamma', nn.softplus(mu))

    def transition_fn(carry, t):
        lam_prev = carry

        U = jnp.log(lam_prev) - jnp.log(lam_prev.sum(-1, keepdims=True))

        logs = logits((beliefs[0][:, t], beliefs[1][:, t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        lam_next = npyro.deterministic('lams', lam_prev + nn.one_hot(beliefs[2][t], 4) * jnp.expand_dims(mask[t] * eta, -1))

        mixing_dist = dist.CategoricalProbs(weights)
        component_dist = dist.CategoricalLogits(logs.swapaxes(0, 1)).mask(mask[t][..., None])
        with npyro.plate('subjects', N):
            npyro.sample('y', dist.MixtureSameFamily(mixing_dist, component_dist))

        return (lam_next), None

    lam_start = npyro.deterministic('lam_start', lam0 + jnp.expand_dims(eta, -1) * c0)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, (lam_start), jnp.arange(T)
        )

def gammadyn_mixture_model(beliefs, y, mask):
    M, T, _ = beliefs[0].shape

    tau = .5
    weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(M)))

    assert weights.shape == (M,)
    
    gamma = npyro.sample('gamma', dist.InverseGamma(2., 2.))
    mu = jnp.log(jnp.exp(gamma) - 1)


    p = npyro.sample('p', dist.Dirichlet(jnp.ones(3)))
    p0 = npyro.deterministic('p0', 
        jnp.concatenate([p[..., :1]/2, p[..., :1]/2 + p[..., 1:2], p[..., 2:]/2, p[..., 2:]/2], -1)
    )

    scale = npyro.sample('scale',  dist.Gamma(1., 1.))
    rho = npyro.sample('rho', dist.Beta(1., 2.))
    sigma = jnp.sqrt( - (1 - rho**2) / (2 * jnp.log(rho)) ) * scale

    U = jnp.log(p0)

    def transition_fn(carry, t):
        x_prev = carry

        gamma_dyn = npyro.deterministic('gamma_dyn', nn.softplus(mu + x_prev))

        logs = logits((beliefs[0][:, t], 
                       beliefs[1][:, t]), 
                       jnp.expand_dims(gamma_dyn, -1), 
                       jnp.expand_dims(U, -2))

        mixing_dist = dist.CategoricalProbs(weights)
        component_dist = dist.CategoricalLogits(logs).mask(mask[t])
        npyro.sample('y', dist.MixtureSameFamily(mixing_dist, component_dist))

        with npyro.handlers.reparam(config={"x_next": npyro.infer.reparam.TransformReparam()}):
            affine = dist.transforms.AffineTransform(rho*x_prev, sigma)
            x_next = npyro.sample('x_next', dist.TransformedDistribution(dist.Normal(0., 1.), affine))

        return (x_next), None

    x0 = jnp.zeros(1)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, (x0), jnp.arange(T)
        )

def nondynamic_mixture_model(beliefs, y, mask):
    M, T, _ = beliefs[0].shape

    tau = .5
    weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(M)))

    assert weights.shape == (M,)

    gamma = npyro.sample('gamma', dist.InverseGamma(2., 5.))
    
    p = npyro.sample('p', dist.Dirichlet(jnp.ones(3)))
    
    p0 = npyro.deterministic('p0', jnp.concatenate([p[..., :1]/2, p[..., :1]/2 + p[..., 1:2], p[..., 2:]/2, p[..., 2:]/2], -1))

    U = jnp.log(p0)

    def transition_fn(carry, t):

        logs = logits((beliefs[0][:, t], beliefs[1][:, t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        mixing_dist = dist.CategoricalProbs(weights)
        component_dist = dist.CategoricalLogits(logs).mask(mask[t])
        npyro.sample('y', dist.MixtureSameFamily(mixing_dist, component_dist))

        return None, None

    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, None, jnp.arange(T)
        )