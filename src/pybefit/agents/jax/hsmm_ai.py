#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of active inference agent for temporal reversal learning task in numpyro.
@author: Dimitrije Markovic
"""
import numpy as np
import jax.numpy as jnp

from jax.scipy.special import digamma
from jax import random, vmap, nn, devices, device_put
from scipy.special import binom

from .utils import einsum, log

vdiag = vmap(lambda v: jnp.diag(v, k=0))
voffd = vmap(lambda v: jnp.diag(v, k=1))

try:
    gpu = devices('gpu')[0]
except:
    gpu = None

devs = {'cpu': devices('cpu')[0], 'gpu': gpu}

def par_efe(p_c, params, U):
    # expected free energy
    p_aco = params/params.sum(-1, keepdims=True)
    q_ao = einsum('...aco,...c->...ao', p_aco, p_c)

    KL_a =  - jnp.sum(q_ao * U, -1) + jnp.sum(q_ao * log(q_ao), -1)

    H_ac = - (p_aco * digamma(params)).sum(-1) + digamma(params.sum(-1) + 1)
    H_a = einsum('...c,...ac->...a', p_c, H_ac)

    return KL_a + H_a
    
def logits(beliefs, gamma, U):
    # expected surprisal based action selection
    p_c, params = beliefs
    S_a = par_efe(p_c, params, U)

    return - gamma * ( S_a - S_a.min(-1, keepdims=True))

class HSMMAI(object):
    def __init__(self, 
                    T, 
                    N, 
                    nu_max=1, 
                    nu_min=0, 
                    U=None,
                    lam_0=None,
                    eta=None,
                    dynamic_preferences=False, 
                    mask=None,
                    prior_kwargs={},  
                    device='cpu'):
        
        self.device = devs[device]
        
        self.T = T
        self.N = N
        self.nu_max = nu_max
        self.nu_min = nu_min

        self.slc = device_put(jnp.arange(self.N), self.device)
        self.dyn_pref = dynamic_preferences

        if self.dyn_pref:
            if lam_0 is None:
                self.lam = jnp.ones((N, 4))
                self.eta = 0.05 * jnp.ones(N)
            else:
                self.lam = lam_0
                self.eta = eta
        else:
            if U is None:
                self.U = device_put(jnp.array([-.5, .5, 0., 0.]), self.device)
            else:
                self.U = device_put(U, self.device)
                assert self.U.shape[-1:] == (4,)
                
        if mask is None:
            self.mask = device_put(jnp.ones((T, N)), self.device)
        else:
            self.mask = device_put(jnp.broadcast_to(mask, (T, N)), self.device)
        
        assert self.mask.shape == (T, N)

        self.logits = None
        
        self.__make_transition_matrices()
        self.__set_prior(**prior_kwargs)
        
    def __make_transition_matrices(self):
        nu = np.arange(1, self.nu_max + 1)
        mu = np.arange(4., 45., 1.)
        p = mu[..., None]/(nu[None] + mu[..., None])
        
        j = np.tril(nu[None].repeat(self.nu_max, 0))
        bnm = binom(nu[..., None], np.tril(j - 1))
        p1 = p[..., None] ** np.tril((nu[..., None] - j + 1))
        p2 = (1 - p[..., None])**np.tril(j - 1)
        pi =  np.tril(bnm * p1 * p2)
        pi = np.concatenate([pi, 1 - pi.sum(-1, keepdims=True)], -1)
        
        # phase transition matrix for different models m, f_t| f_{t+1}
        p_ff = []
        for i in range(self.nu_max):
            vp = p[..., i:i + 1].repeat(i + 1, -1)
            tmp = np.concatenate([vdiag(vp) + voffd(1 - vp[..., :-1]), np.zeros((p.shape[0], i + 1, self.nu_max - i - 1))], -1)
            tmp = np.concatenate([tmp, 1 - tmp.sum(-1, keepdims=True)], -1)
            tmp = np.concatenate([tmp, np.ones((p.shape[0], self.nu_max - i - 1, self.nu_max + 1))/(self.nu_max + 1)], -2)
            tmp = np.concatenate([tmp, pi[:, i:i+1]], -2)
            p_ff.append(tmp)
        
        self.p_mff = device_put(jnp.stack(p_ff, -3).reshape(-1, self.nu_max + 1, self.nu_max + 1), self.device)
        
        # state transition matrix f_t, j_t| j_{t+1}
        p_fjj = np.zeros((self.nu_max + 1, 2, 2))
        p_fjj[-1, :, 1] = 1.
        p_fjj[:-1, :, 0] = 1.
        
        p_jcc = np.stack([np.eye(2), (np.ones((2, 2)) - np.eye(2))], 0)

        self.p_fcc = einsum('jcz,fj->fcz', device_put(p_jcc, self.device), device_put(p_fjj[:, 0], self.device))
        
    def __set_prior(self, a=6., b=32.):
        prior_mf = self.p_mff[:, -1]
        M = prior_mf.shape[0]
        prior_m = np.ones(M)/M
        prior_m = prior_m.reshape(-1, self.nu_max)
        prior_m = np.concatenate([np.zeros_like(prior_m[:, :self.nu_min]), prior_m[:, self.nu_min:]], -1).reshape(-1)
        prior_m /= prior_m.sum()
        prior_fm = (prior_mf * device_put(prior_m[:, None], self.device)).T
        prior_c = device_put(jnp.ones(2)/2, self.device)

        pars = device_put(jnp.array([
            [[a, b, 1, 1], [b, a, 1, 1]],
            [[b, a, 1, 1], [a, b, 1, 1]],
            [[1, 1, 1000., 1], [1, 1, 1, 1000.]]
        ]), self.device)[None].repeat(self.N, 0)
        
        probs = einsum('c,fm->cfm', prior_c, prior_fm)[None].repeat(self.N, 0)
        self.prior = (probs, pars)
        
    def action_selection(self, rng_key, beliefs, gamma=1e3):
        # sample choices

        p_cfm, params = beliefs

        p_c = einsum('...cfm->...c', p_cfm)

        beliefs = (p_c, params)

        if self.dyn_pref:
            U = jnp.expand_dims(jnp.log(self.lam), -2)
        else:
            U = self.U

        self.logits = logits(beliefs, gamma, U)

        return random.categorical(rng_key, self.logits)

    def learning(self, t, observations, responses, prior):
        # parametric learning and inference
        p_cfm, params = prior
        obs = nn.one_hot(observations, 4)

        p_aco = params/params.sum(-1, keepdims=True)

        p_c = einsum('nco,no->nc', p_aco[self.slc, responses], obs)
        
        m = jnp.expand_dims(self.mask[t], -1)
        p_c = m * p_c + (1 - m)/2.

        post = einsum('nc,ncfm->ncfm', p_c, p_cfm)
        
        norm = post.reshape(post.shape[:-3] + (-1,)).sum(-1)[..., None, None, None]
        post = post/norm

        resp = nn.one_hot(responses, 3)
        post_c = post.reshape(post.shape[:-2] + (-1,)).sum(-1)

        params_new = params + einsum('na,nc,no,n->naco', resp, post_c, obs, self.mask[t])
        pred = einsum('fcz,mfg,ncfm->nzgm', self.p_fcc, self.p_mff, post)

        if self.dyn_pref:
            self.lam += jnp.expand_dims(self.eta, -1) * obs

        return (pred, params_new)
