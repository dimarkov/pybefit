#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:14:01 2019

This module contains RL agents using backward infuction to 
estimate optimal policy in various experimental tasks. 

@author: Dimitrije Markovic
"""

import torch
from torch import zeros, ones, tensor, eye

from torch.distributions import Categorical, Uniform

from numpy import nan_to_num, indices, ravel_multi_index

from itertools import product

from .agent import Discrete

__all__ = [
        'BIBanditsFlat',
        'BIBanditsDeep'
]

def ntn(x):
    return torch.nan_to_num(x)

def mli(x, L):
    return ravel_multi_index(x, (L,)*2)

def psi(x):
    return torch.digamma(x)

class BIBanditsFlat(Discrete):
    '''Agent using backward induction to compute optimal choice.
    '''
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        self.nit = 10
        
        na = pars['na']  # number of choices
        self.nl = pars['nl']  # number of arms
        self.nf = pars['nf']  # number of features
        self.ni = pars['ni']  # number of internal states
        
        super(BIBanditsFlat, self).__init__(runs, blocks, trials, na, None, None)
        
        self.priors = {}
                
        self.initiate_policies()
        self.initiate_prior_beliefs()
        self.initiate_beliefs_and_expectations()
                
        
    def initiate_beliefs_and_expectations(self):
        
        self.beliefs = { 
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.nl),  # selected location/ arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf, dtype=torch.long),
                }
        
        self.expectations = {}

    def set_parameters(self, x=None, priors=None, set_variables=True):
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)
        
        if x is None:
            self.a = [ones(self.runs, self.na, self.nf)]
        else:
            self.a = [x[-1]]
            
        if priors is not None:
            self.initiate_prior_beliefs(**priors)
        else:
            self.initiate_prior_beliefs()
            
        self.eps = 1/20
        
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        self.logits = []
        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('locations', [])
        self.priors.setdefault('policies', [])

        self.priors['locations'].append(ones(self.na)/self.na)
        self.priors['policies'].append(ones(self.npi)/self.npi)
        
    
    def update_prior_beliefs(self, b, t, **kwargs):
        
        a = self.a[-1]
        a0 = a.sum(-1, keepdim=True)
        
        A = a/a0
        
        # increase uncertainty between segments but keep expectations the same
        if b > 0:
            v = a0*(1 - self.eps) + self.eps
            self.a.append(v*A)
        
        self.update_observation_likelihood()
    
    def update_observation_likelihood(self):
        a = self.a[-1]
        a0 = self.a[-1].sum(-1, keepdim=True)
        
        self.A = a/a0
        
    def update_beliefs(self, b, t, response_outcomes):
        res = response_outcomes[-2]
        out1, out2 = response_outcomes[-1]
        
        a = self.a[-1].clone()
        
        a[range(self.runs), res] += out1
        
        self.a.append(a)
        
        self.beliefs['locations'][b, t] = torch.eye(self.na)[res]
        
        self.update_observation_likelihood()
    
    def backward_induction(self, b, t):
        runs = self.runs
        nt = self.nt
        
        mpt = nt  # maximum number of points
        np = self.nf - 1  # number of different types of points
        
        thr = 2*mpt//3  # success threshold
        
        # predictions
        L = torch.tensor(list(product(range(self.nf), repeat= nt-t)))
        bonus1 = torch.sum(L == 1, -1)
        bonus2 = torch.sum(L == 2, -1)
        
        actions = torch.arange(self.na)
        log_probs = 0.
        for k in range(t+1, nt+1):
            actions = self.policies[:, k-1]
            
            outcomes = self.A[:, actions]
            log_probs += outcomes[..., L[:, k - t - 1]].log()
            
        probs = log_probs.exp()
        p1 = self.beliefs['points'][b, t][:, 1].reshape(-1, 1) + bonus1.reshape(1, -1)
        p2 = self.beliefs['points'][b, t][:, 2].reshape(-1, 1) + bonus2.reshape(1, -1)
        
        l = (p1 <= thr)*(p2 <= thr)
        
        Q = zeros(runs, self.na)
        actions = self.policies[:, t]
        for n in range(runs):
            for a in range(self.na):
                r = 1 - probs[n, :, l[n]].sum(-1)
                Q[n, a] = r[a == actions].max()
        
        return Q
        
    def planning(self, b, t, **kwargs):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        
        if t == 0:
            self.update_prior_beliefs(b, t)
            
        if 'locations' in kwargs:
            locations = kwargs['locations']
            self.beliefs['locations'][b, t] = 0.
            self.beliefs['locations'][b, t, range(self.runs), locations] = 1.
        
        if 'points' in kwargs:
            points = kwargs['points']
            self.beliefs['points'][b, t] = points
        
        Q = self.backward_induction(b, t)        
    
        self.logits.append(self.alpha.reshape(-1, 1)*(2*Q-1))   
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()
    

class BIBanditsDeep(Discrete):
    '''Agent using backward induction to compute optimal choice.
    '''
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        self.nit = 10
        
        na = pars['na']  # number of choices
        no = pars['no']  # number of offers
        self.nf = pars['nf']  # number of features
        self.nc = pars['nc']  # number of contexts
        self.ni = pars['ni']  # number of internal states
        
        super(BIBanditsDeep, self).__init__(runs, blocks, trials, na, None, no)
                
        self.priors = {}
        
        self.initiate_policies()
        self.initiate_beliefs_and_expectations()
        
    def initiate_beliefs_and_expectations(self):
        
        self.beliefs = { 
                'context': zeros(self.nb, self.runs, self.nc),  # context states 2nd level
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states 2nd level
                'offers': zeros(self.nb, self.nt + 1, self.runs, self.no),  # offer state
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.na),  # selected location/ arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf, dtype=torch.long),
                }

        self.expectations = {}


    def set_parameters(self, x=None, priors=None, set_variables=True):
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)
        
        if x is None:
            self.a = [ones(self.runs, self.no, self.na, self.nf)]
        else:
            self.a = [x[-1]]
            
        if priors is not None:
            self.initiate_prior_beliefs(**priors)
        else:
            self.initiate_prior_beliefs()
        
        # context dependent offer transitions
        self.cBo = eye(self.no).repeat(self.nc, 1, 1)
        M = (ones(self.no, self.no) - eye(self.no)).repeat(self.nc, 1, 1)/(self.no-1)
        rho = 1.
        self.cBo = rho*self.cBo + (1-rho)*M
        
        rho2 = .95
        M = (torch.diag(ones(self.nc-1), 1) + torch.diag(ones(self.nc-2), -2))/(self.nc-1)
        self.Bc = rho2*eye(self.nc) + (1-rho2)*M 
        
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        self.logits = []
        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('context', [])
        self.priors.setdefault('offers', [])
        self.priors.setdefault('locations', [])
        self.priors.setdefault('probs', [])
        self.priors.setdefault('policies', [])

        if 'context' in kwargs:
            self.priors['context'].append(kwargs['context'])
        else:
            self.priors['context'].append(torch.tensor([1., 0., 0.]).repeat(self.runs, 1))
        
        self.priors['offers'].append(2*eye(self.nc).repeat(self.runs, 1, 1) + 1)
        self.priors['locations'].append(ones(self.na)/self.na)
        self.priors['policies'].append(ones(self.npi)/self.npi)
    
    def update_state_transition_matrix(self):
        prior = self.priors['context'][-1]
        self.Bo = torch.einsum('nj,jkl->nkl', prior, self.cBo)
       
    def update_observation_likelihood(self):
        a = self.a[-1]
        a0 = self.a[-1].sum(-1, keepdim=True)
        
        self.A = a/a0
        
    def update_prior_beliefs(self, b, t, **kwargs):
        if 'context' in kwargs:
            cnt = kwargs['context']
            context = torch.eye(self.nc)[cnt]
            self.priors['context'].append(context)
        else:
            context = self.priors['context'][-1]
        
        f = self.priors['offers'][-1]
        f0 = f.sum(-1, keepdim=True)
        
        self.D = f/f0
        
        # set prior over offers as a marginal over contexts
        self.beliefs['offers'][b, t] = torch.einsum('ni,nij->nj', context, self.D)
        
        self.update_state_transition_matrix()
        self.update_observation_likelihood()
                
    def update_second_level(self, b, t, out1, out2):
        
        context = self.priors['context'][-1]
        offers = self.beliefs['offers'][b, t]
        locations = self.beliefs['locations'][b, t]
        
        alpha = self.beliefs['offers'][b, :t+1]
        gamma = self.beliefs['offers'][b, :t+2].clone()
        tm = self.Bo
        for k in range(t+1):
            pred = torch.einsum('ni,nij->nj', alpha[-k-1], tm)
            gamma[-k-2] = torch.einsum('nj,nij,ni->ni', gamma[-k-1]/pred, tm, alpha[-k-1])
        
        cp = self.priors['context'][-1]
        op = self.priors['offers'][-1]/self.priors['offers'][-1].sum(-1, keepdim=True)
        context = (torch.einsum('nco,no->nc', op.log(), gamma[0]) + cp.log()).softmax(-1)
        
        f = torch.einsum('nc,nf->ncf', context, gamma[0]) + self.priors['offers'][-1]  
        a = torch.einsum('no,nl,nf->nolf', offers, locations, out1) + self.a[-1]
        
        self.a.append(a)
        self.priors['offers'].append(f)
        self.update_observation_likelihood()
        
        if t == self.nt-1:
            self.priors['context'].append(context@self.Bc)
    
    def update_beliefs(self, b, t, response_outcomes):
        res = response_outcomes[-2]
        out1, out2 = response_outcomes[-1]
        
        offers = self.beliefs['offers'][b, t]
        
        A = self.A[range(self.runs), :, :, out1.argmax(-1)]
        post = torch.einsum('ni,nij->nij', offers, A)[range(self.runs), :, res]
        post /= post.sum(-1, keepdim=True)
        
        self.beliefs['offers'][b, t] = post
        self.beliefs['offers'][b, t + 1] = torch.einsum('nij,ni->nj', self.Bo, self.beliefs['offers'][b, t])
        
        self.beliefs['locations'][b, t] = torch.eye(self.na)[res]
        
        self.update_second_level(b, t, out1, out2)
    
    def backward_induction(self, b, t):
        runs = self.runs
        nt = self.nt
        
        mpt = nt  # maximum number of points
        np = self.nf - 1  # number of different types of points
        
        thr = 2*mpt//3  # success threshold
        
        # predictions
        L = torch.tensor(list(product(range(self.nf), repeat= nt-t)))
        bonus1 = torch.sum(L == 1, -1)
        bonus2 = torch.sum(L == 2, -1)
        
        actions = torch.arange(self.na)
        log_probs = 0.
        for k in range(t+1, nt+1):
            actions = self.policies[:, k-1]
            
            beliefs = self.beliefs['offers'][b, k-1]
            predictions = torch.einsum('ni,nij->nj', beliefs, self.Bo)
            self.beliefs['offers'][b, k] = predictions

            outcomes = torch.einsum('ni,nijk->njk', predictions, self.A)[:, actions]
            log_probs += outcomes[..., L[:, k - t - 1]].log()
            
        probs = log_probs.exp()
        p1 = self.beliefs['points'][b, t][:, 1].reshape(-1, 1) + bonus1.reshape(1, -1)
        p2 = self.beliefs['points'][b, t][:, 2].reshape(-1, 1) + bonus2.reshape(1, -1)
        
        l = (p1 <= thr)*(p2 <= thr)
        
        Q = zeros(runs, self.na)
        actions = self.policies[:, t]
        for n in range(runs):
            for a in range(self.na):
                r = 1 - probs[n, :, l[n]].sum(-1)
                Q[n, a] = r[a == actions].max()
            
        return Q
        
    def planning(self, b, t, **kwargs):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if 'context' in kwargs:
            self.update_prior_beliefs(b, t, **kwargs)
        elif t == 0:
            self.update_prior_beliefs(b, t)
        
        if 'locations' in kwargs:
            locations = kwargs['locations']
            self.beliefs['locations'][b, t] = 0.
            self.beliefs['locations'][b, t, range(self.runs), locations] = 1.
        
        if 'points' in kwargs:
            points = kwargs['points']
            self.beliefs['points'][b, t] = points
        
        if t == 0:
            self.update_state_transition_matrix()
        
        Q = self.backward_induction(b, t)        
    
        self.logits.append(self.alpha.reshape(-1, 1)*(2*Q-1))   
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()


