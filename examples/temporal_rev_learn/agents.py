#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of active inference agent for temporal reversal learning task 
in numpyro
@author: Dimitrije Markovic
"""

import jax.numpy as jnp

from jax.scipy.special import digamma
from jax import nn, random, lax, vmap, ops

from scipy.special import binom
from opt_einsum import contract

vdiag = vmap(lambda v: jnp.diag(v, k=0))
voffd = vmap(lambda v: jnp.diag(v, k=1))

def einsum(equation, *args):
    return contract(equation, *args, backend='jax')

def log(x, minimum=-1e10):
    return jnp.where(x > 0, jnp.log(x), minimum) 


class Agent(object):
    def __init__(self, N, nu_max=1, nu_min=0):
        self.N = N
        self.nu_max = nu_max
        self.nu_min = nu_min
        
        self.__make_transition_matrices()
        self.__set_prior()
        
    def __make_transition_matrices(self):
        nu = jnp.arange(1, self.nu_max + 1)
        mu = jnp.arange(4., 45., 1.)
        p = mu[..., None]/(nu[None] + mu[..., None])
        
        j = jnp.tril(jnp.arange(1, self.nu_max + 1)[None].repeat(self.nu_max, 0))
        bnm = binom(nu[..., None], jnp.tril(j-1))
        p1 = p[..., None] ** jnp.tril((nu[..., None] - j + 1))
        p2 = (1 - p[..., None])**jnp.tril(j - 1)
        pi =  jnp.tril(bnm * p1 * p2)
        pi = jnp.concatenate([pi, 1 - pi.sum(-1, keepdims=True)], -1)
        
        # phase transition matrix for different models m, f_t| f_{t+1}
        p_ff = []
        for i in range(self.nu_max):
            vp = p[..., i:i + 1].repeat(i + 1, -1)
            tmp = jnp.concatenate([vdiag(vp) + voffd(1 - vp[..., :-1]), jnp.zeros((p.shape[0], i + 1, self.nu_max - i - 1))], -1)
            tmp = jnp.concatenate([tmp, 1 - tmp.sum(-1, keepdims=True)], -1)
            tmp = jnp.concatenate([tmp, jnp.ones((p.shape[0], self.nu_max - i - 1, self.nu_max + 1))/(self.nu_max + 1)], -2)
            tmp = jnp.concatenate([tmp, pi[:, i:i+1]], -2)
            p_ff.append(tmp)
        
        self.p_mff = jnp.stack(p_ff, -3).reshape(-1, self.nu_max + 1, self.nu_max + 1)
        
        # state transition matrix f_t, j_t| j_{t+1}
        p_fjj = jnp.zeros((self.nu_max + 1, 2, 2))
        p_fjj = ops.index_add(p_fjj, ops.index[-1, :, 1], 1.)
        p_fjj = ops.index_add(p_fjj, ops.index[:-1, :, 0], 1.)
        
        p_jcc = jnp.stack([jnp.eye(2), (jnp.ones((2, 2)) - jnp.eye(2))], 0)

        self.p_fcc = einsum('jcz,fj->fcz', p_jcc, p_fjj[:, 0])
        
    def __set_prior(self, a=6., b=32.):
        prior_mf = self.p_mff[:, -1]
        M = prior_mf.shape[0]
        prior_m = jnp.ones(M)/M
        prior_m = prior_m.reshape(-1, self.nu_max)
        prior_m = jnp.concatenate([jnp.zeros_like(prior_m[:, :self.nu_min]), prior_m[:, self.nu_min:]], -1).reshape(-1)
        prior_m /= prior_m.sum()
        prior_fm = (prior_mf * prior_m[:, None]).T
        prior_c = jnp.ones(2)/2

        pars = jnp.array([
            [[a, b, 1, 1], [b, a, 1, 1]],
            [[b, a, 1, 1], [a, b, 1, 1]],
            [[1, 1, 1000., 1], [1, 1, 1, 1000.]]
        ])[None].repeat(self.N, 0)
        
        probs = einsum('c,fm->cfm', prior_c, prior_fm)[None].repeat(self.N, 0)
        self.prior = (probs, pars)
        
    def __par_efe(self, p_c, params, U):
        p_aco = params/params.sum(-1, keepdims=True)
        q_ao = einsum('...aco,...c->...ao', p_aco, p_c)
    
        KL_a =  - jnp.sum(q_ao * U, -1) + jnp.sum(q_ao * log(q_ao), -1)
    
        H_ac = - (p_aco * digamma(params)).sum(-1) + digamma(params.sum(-1) + 1)
        H_a = einsum('...c,...ac->...a', p_c, H_ac)
    
        return KL_a + H_a
    
    def logits(self, beliefs, gamma, U):
        p_cfm, params = beliefs

        # expected surprisal based action selection
        p_c = einsum('...cfm->...c', p_cfm)

        S_a = self.__par_efe(p_c, params, U)

        return - gamma * ( S_a - S_a.min(-1, keepdims=True))

    def action_selection(self, rng_key, beliefs, gamma=1e3, U=jnp.array([-1., 1., 0., 0.])):
        # sample choices
        return random.categorical(rng_key, self.logits(beliefs, gamma, U))

    def learning(self, observations, responses, prior, mask=jnp.ones(1)):
        # parametric learning and inference
        p_cfm, params = prior
        obs = jnp.eye(4)[observations]

        p_aco = params/params.sum(-1, keepdims=True)

        p_c = einsum('nco,no->nc', p_aco[jnp.arange(self.N), responses], obs)
        
        m = jnp.expand_dims(mask, -1)
        p_c = m * p_c + (1 - m)/2.

        post = einsum('nc,ncfm->ncfm', p_c, p_cfm)
        
        norm = post.reshape(post.shape[:-3] + (-1,)).sum(-1)[..., None, None, None]
        post = post/norm

        resp = jnp.eye(3)[responses]
        post_c = post.reshape(post.shape[:-2] + (-1,)).sum(-1)

        params_new = params + einsum('na,nc,no,n->naco', resp, post_c, obs, mask)
        pred = einsum('fcz,mfg,ncfm->nzgm', self.p_fcc, self.p_mff, post)

        return (pred, params_new)
