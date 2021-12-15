#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for simulations and inference
@author: Dimitrije Markovic
"""

from jax.interpreters.batching import batch
import jax.numpy as jnp
import numpyro as npyro
import numpyro.distributions as dist
from jax import random, lax, nn
from numpyro.infer import log_likelihood
from numpyro.distributions import TransformedDistribution, transforms
from agents import Agent, logits
from opt_einsum import contract
from numpyro.contrib.control_flow import scan

def einsum(equation, *args):
    return contract(equation, *args, backend='jax')

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

def estimate_beliefs(outcomes, choices, mask=1, nu_max=10, nu_min=0, **kwargs):
    # belief estimator from fixed responses and outcomes
    T, N = choices.shape
    assert outcomes.shape == (T, N)
    agent = Agent(T, N, nu_max=nu_max, nu_min=nu_min, mask=mask, prior_kwargs=kwargs)
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
    log_lk = log_likelihood(model, samples, *args, **kwargs)['y'].sum(-2)
    S = log_lk.shape[0]
    lppd = nn.logsumexp(log_lk, 0) - jnp.log(S)
    p_waic = jnp.var(log_lk, axis=0, ddof=1)
    return {'lpd': lppd, 'waic': lppd - p_waic}


from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import validate_sample 

class CategoricalMixture(Distribution):
    arg_constraints = {'weights': constraints.simplex}
    has_enumerate_support = False
    is_discrete = True

    def __init__(self, component_distribution, weights, validate_args=None):
        self.component_distribution = component_distribution
        self.weights = weights
        
        batch_shape = weights.shape[:-1]
        assert component_distribution.batch_shape[:-1] == batch_shape

        super().__init__(
            batch_shape=batch_shape, 
            event_shape=component_distribution.event_shape, 
            validate_args=validate_args
        )
    
    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    @property
    def mixture_dim(self):
        return -self.event_dim - 1 

    @validate_sample
    def log_prob(self, value):

        value = jnp.expand_dims(value, self.mixture_dim)
        component_log_probs = self.component_distribution.log_prob(value)

        sum_log_probs = jnp.log(self.weights) + component_log_probs
        log_prob = nn.logsumexp(sum_log_probs, -1)  

        return log_prob

    @property
    def support(self):
        return self.component_distribution.support

def mixture_model(sequences, y, mask):
    M, T, N, _ = sequences[0].shape
    
    tau = npyro.sample('tau', dist.HalfCauchy(1.))
    with npyro.plate('N', N):
        weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(M)))

        gamma = npyro.sample('gamma', dist.Gamma(20., 2.))

        td = TransformedDistribution(
            dist.Normal(jnp.array([-1., .7]), jnp.array([1., 0.2])).to_event(1), 
            transforms.OrderedTransform()
        )

        lams = npyro.sample('lams', td)
        U = jnp.pad(lams, ((0, 0), (0, 2)), 'constant', constant_values=(0,))

        logs = logits(sequences, jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))
        npyro.sample('obs', CategoricalMixture(logs, weights, mask), obs=y)


def mixture_model_dynpref(beliefs, y, mask):
    M, T, N, _ = beliefs[0].shape

    c0 = beliefs[-1]
    
    tau = npyro.sample('tau', dist.HalfCauchy(1.))
    dt = 1. # npyro.sample('dt', dist.HalfNormal(1.))
    with npyro.plate('N', N):
        weights = npyro.sample('weights', dist.Dirichlet(tau * jnp.ones(M)))

        assert weights.shape == (N, M)

        x0 = npyro.sample('x0', dist.Normal(0., 1.))
        scale = npyro.sample('scale', dist.HalfCauchy(1.))
        theta = npyro.sample('theta', dist.HalfCauchy(1.))
        mu = npyro.sample('mu', dist.Normal(5., 5.))

        lam12 = npyro.sample('lam12', dist.HalfCauchy(1.).expand([2]).to_event(1))
        lam34 = npyro.sample('lam34', dist.HalfCauchy(1.).expand([2]).to_event(1))

        lam0 = npyro.deterministic('lam0', jnp.concatenate([lam12.cumsum(-1), lam34], -1))
        eta = npyro.sample('eta', dist.Beta(1, 10))

        rho = jnp.exp(- dt * theta)
        sigma = jnp.sqrt( (1 - rho**2) / (2 * theta) ) * scale

    def transition_fn(carry, t):
        lam_prev, x_prev = carry

        gamma = npyro.deterministic('gamma', nn.softplus(mu + x_prev))

        U = jnp.log(lam_prev)

        logs = logits((beliefs[0][:, t], beliefs[1][:, t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))

        lam_next = npyro.deterministic('lams', lam_prev + nn.one_hot(beliefs[2][t], 4) * jnp.expand_dims(mask[t] * eta, -1))

        mixing_dist = dist.CategoricalProbs(weights)
        component_dist = dist.CategoricalLogits(logs.swapaxes(0, 1)).mask(mask[t][..., None])
        with npyro.plate('subjects', N):
            y = npyro.sample('y', dist.MixtureSameFamily(mixing_dist, component_dist))
            noise = npyro.sample('dw', dist.Normal(0., 1.))

        x_next = rho * x_prev + sigma * noise

        return (lam_next, x_next), None

    lam_start = npyro.deterministic('lam_start', lam0 + eta[:, None] * c0)
    with npyro.handlers.condition(data={"y": y}):
        scan(
            transition_fn, (lam_start, x0), jnp.arange(T)
        )

def dynamic_preference_gm(beliefs, y=None, mask=True):
    # generative model
    T, N = beliefs[0].shape[:2]
    mask = jnp.broadcast_to(mask, (T, N))

    c0 = beliefs[-1]

    dt = 1.
    with npyro.plate('N', N):
        x0 = npyro.sample('x0', dist.Normal(0., 1.))
        scale = npyro.sample('scale', dist.HalfCauchy(1.))
        theta = npyro.sample('theta', dist.HalfCauchy(1.))
        mu = npyro.sample('mu', dist.Normal(5., 5.))

        lam12 = npyro.sample('lam12', dist.HalfCauchy(1.).expand([2]).to_event(1))
        lam34 = npyro.sample('lam34', dist.HalfCauchy(1.).expand([2]).to_event(1))

        lam0 = npyro.deterministic('lam0', jnp.concatenate([lam12.cumsum(-1), lam34], -1))

        eta = npyro.sample('eta', dist.Beta(1, 10))

        rho = jnp.exp(- dt * theta)
        sigma = jnp.sqrt( (1 - rho**2) / (2 * theta) ) * scale

    def transition_fn(carry, t):
        lam_prev, x_prev = carry
        
        gamma = npyro.deterministic('gamma', nn.softplus(mu + x_prev))
        U = jnp.log(lam_prev)
        logs = npyro.deterministic(
            'logits',
            logits((beliefs[0][t], beliefs[1][t]), jnp.expand_dims(gamma, -1), jnp.expand_dims(U, -2))
        )
        with npyro.plate('subjects', N):
            y = npyro.sample('y', dist.CategoricalLogits(logs).mask(mask[t]))
            noise = npyro.sample('dw', dist.Normal(0., 1.))

        lam_next = npyro.deterministic('lam', lam_prev + nn.one_hot(beliefs[2][t], 4) * jnp.expand_dims(mask[t] * eta, -1))
        
        x_next = rho * x_prev + sigma * noise

        return (lam_next, x_next), (y, x_next)

    lam_start = npyro.deterministic('lam_start', lam0 + eta[:, None] * c0)

    with npyro.handlers.condition(data={"y": y}):
        _, ys = scan(
            transition_fn, (lam_start, x0), jnp.arange(T)
        )