#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:14:01 2019

This module contains active inference agents for various experimental tasks. 

@author: Dimitrije Markovic
"""

import torch
from torch import zeros, ones, tensor, eye

from torch.distributions import Categorical, Uniform

from numpy import nan_to_num

from itertools import product

from .agent import Discrete

__all__ = [
        'BIBandits'
]

def ntn(x):
    return torch.from_numpy(nan_to_num(x))

class BIBandits(Discrete):
    '''Agent using backward induction to compute optimal choice.
    '''
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        self.nit = 10
        
        na = pars['na']  # number of choices
        ns = pars['ns']  # number of states
        no = pars['no']  # number of offers
        self.nl = pars['nl']  # number of arms
        self.nf = pars['nf']  # number of features
        self.nc = pars['nc']  # number of contexts
        self.ni = pars['ni']  # number of internal states
        
        super(BIBandits, self).__init__(runs, blocks, trials, na, ns, no)
                
        self.pr_arms, self.arm_types = pars['arm-types']  # define arm type for each offer-choice pair
        
        self.priors = {}
        
        self.set_policies()
        self.initiate_beliefs()
        
        self.G = zeros(self.nb, self.nt, self.runs, self.npi)
        self.eU = zeros(self.nb, self.nt, self.runs, self.npi)
        self.eH = zeros(self.nb, self.nt, self.runs, self.npi)
        self.eE = zeros(self.nb, self.nt, self.runs, self.npi)
        
        self.gamma = []
                
        
    def initiate_beliefs(self):
        
        self.beliefs = { 
                'context': zeros(self.nb, self.runs, self.nc),  # context states 2nd level
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states 2nd level
                'offers': zeros(self.nb, self.nt + 1, self.runs, self.no),  # offer state
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.nl),  # selected location/ arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf, dtype=torch.long),
                }

    def set_parameters(self, x=None, priors=None, set_variables=True):
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)
        
        if x is None:
            self.a = [ones(self.runs, self.nc, self.no, self.nl, self.nf)]
        else:
            self.a = [x[-1]]
            
        if priors is not None:
            self.set_prior_beliefs(**priors)
        else:
            self.set_prior_beliefs()
        
        # context dependent offer transitions
        self.cBo = eye(self.no).repeat(self.nc, 1, 1)
        M = (ones(self.no, self.no) - eye(self.no)).repeat(self.nc, 1, 1)/(self.no-1)
        rho = 1.
        self.cBo = rho*self.cBo + (1-rho)*M
        
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        utility = torch.tensor([-1., 1., 1.]).repeat(self.runs, 1)
        
        N = self.runs//3
        utility[N:2*N, 1] = -1.
        utility[2*N:, -1] = -1.
        self.U = [utility, torch.tensor([-1., 1., 1.])]
        
        self.w = zeros(self.nb, self.nt, self.runs)

        self.offers = []
        self.logits = []
        
    def set_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
    
    def set_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('context', [])
        self.priors.setdefault('offers', [])
        self.priors.setdefault('locations', [])
        self.priors.setdefault('probs', [])
        self.priors.setdefault('policies', [])

        if 'context' in kwargs:
            self.priors['context'].append(kwargs['context'])
        else:
            self.priors['context'].append(ones(self.runs, self.nc)/self.nc)
        
        self.priors['offers'].append(ones(self.runs, self.nc, self.no)/self.no)
        self.priors['locations'].append(ones(self.nl)/self.nl)
        self.priors['probs'].append(ones(self.nc, self.ns, self.nf)/self.nf)
        self.priors['policies'].append(ones(self.nc, self.npi)/self.npi)
    
    def set_prior_expectations(self, b, t, **kwargs):
        if 'context' in kwargs:
            cnt = kwargs['context']
            context = torch.eye(self.nc)[cnt]
            self.priors['context'].append(context)
        else:
            context = self.priors['context'][-1]
        
        offers = self.priors['offers'][-1]
        self.beliefs['offers'][b, t] = torch.einsum('ni,nij->nj', context, offers)
        
        self.update_state_transition_matrix()
        self.update_observation_likelihood()

    def update_state_transition_matrix(self):
        self.B = zeros(self.na, self.ns, self.ns)
        
        prior = self.priors['context'][-1]
        self.Bo = torch.einsum('nj,jkl->nkl', prior, self.cBo)
       
    def update_observation_likelihood(self):
        context = self.priors['context'][-1]
        
        a = self.a[-1]
        a0 = self.a[-1].sum(-1, keepdim=True)
        
        A = a/a0
        
        self.A = torch.einsum('ni,nijkl->njkl', context, A)
        
    def update_second_level(self, b, t, out1, out2):
        
        context = self.priors['context'][-1]
        offers = self.beliefs['offers'][b, t]
        locations = self.beliefs['locations'][b, t]
        
        a = torch.einsum('ni,nj,nk,nl->nijkl', context, offers, locations, out1)
        
        self.a.append(self.a[-1] + a)
        self.update_observation_likelihood()
    
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
            self.set_prior_expectations(b, t, **kwargs)
        
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


