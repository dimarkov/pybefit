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
        'AIBandits'
]

def ntn(x):
    return torch.from_numpy(nan_to_num(x))

def psi(x):
    return torch.digamma(x)

class AIBandits(Discrete):
    
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        self.nit = 10
        
        na = pars['na']  # number of choices
        ns = pars['ns']  # number of states
        no = pars['no']  # number of offers
        self.nl = pars['nl']  # number of arms
        self.nf = pars['nf']  # number of features
        self.nc = pars['nc']  # number of contexts
        self.ni = pars['ni']  # number of internal states
        
        super(AIBandits, self).__init__(runs, blocks, trials, na, ns, no)
                
        self.priors = {}
        
        self.set_parameters()

        self.initiate_prior_beliefs()
        self.initiate_beliefs_and_expectations()
        
    def initiate_beliefs_and_expectations(self):
        
        self.beliefs = { 
                'context': zeros(self.nb, self.runs, self.nc),  # context states 2nd level
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states 2nd level
                'outcomes2': None,  # outcomes at the second level
                'offers': zeros(self.nb, self.nt + 1, self.runs, self.no),  # offer state first level
                'outcomes': zeros(self.nb, self.nt + 1, self.runs, self.npi, self.nf),  # outcomes first level
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.nl),  # selected location/ arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf-1, self.nt + 1),
                'policies': zeros(self.nb, self.nt, self.runs, self.npi)
                }
        
        # expectations over policies
        self.expectations = {
                'utility': zeros(self.nb, self.nt, self.nt, self.runs, self.npi, 3),
                'ambiguity': zeros(self.nb, self.nt, self.nt, self.runs, self.npi),
                'entropy': zeros(self.nb, self.nt, self.nt, self.runs, self.npi),
                'EFE': zeros(self.nb, self.nt, self.runs, self.npi, 3)
                }

    def set_parameters(self, x=None, normalise=True, depth=1, lam=.1):
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)/10
        
        self.depth = depth
        
        if x is None:
            self.a = [ones(self.runs, self.no, self.nl, self.nf)]
        else:
            self.a = [x[-1]]
        
        # context dependent offer transitions
        self.cBo = eye(self.no).repeat(self.nc, 1, 1)
        M = (ones(self.no, self.no) - eye(self.no)).repeat(self.nc, 1, 1)/(self.no-1)
        rho1 = 1.
        self.cBo = rho1*self.cBo + (1-rho1)*M
        
        rho2 = .95
        M = (torch.diag(ones(self.nc-1), 1) + torch.diag(ones(self.nc-2), -2))/(self.nc-1)
        self.Bc = rho2*eye(self.nc) + (1-rho2)*M 
        
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        utility = torch.tensor([-1., 1., 1.]).repeat(self.runs, 3, 1)
        utility[:, 1, 1] = -1.
        utility[:, -1, -1] = -1.
        if normalise:
            utility -= torch.logsumexp(utility, -1, keepdim=True)
        
        self.U = utility*lam
        
        self.w = zeros(self.nb, self.nt, self.runs)
        self.v = zeros(self.nb, self.nt, self.runs, 3)
        
        self.logits = []  # log choice probabilities are stored here
        
        self.initiate_policies()

        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('context', [])
        self.priors.setdefault('offers', [])
        self.priors.setdefault('locations', [])
        self.priors.setdefault('probs', [])
        self.priors.setdefault('policies', [])
        self.priors.setdefault('internals', [])

        if 'context' in kwargs:
            self.priors['context'].append(kwargs['context'])
        else:
            self.priors['context'].append(torch.tensor([1., 0., 0.]).repeat(self.runs, 1))
        
        self.priors['offers'].append(2*eye(self.nc).repeat(self.runs, 1, 1) + 1)
        self.priors['locations'].append(ones(self.nl)/self.nl)
        self.priors['probs'].append(ones(self.nc, self.ns, self.nf)/self.nf)
        self.priors['policies'].append(ones(self.npi)/self.npi)
        
        self.priors['internals'].append(torch.tensor([0.0000005, .999999/2, .999999/2]).repeat(self.runs, 1))
        
    def update_state_transition_matrix(self):
        self.B = zeros(self.na, self.ns, self.ns)
        
        prior = self.priors['context'][-1]
        self.Bo = torch.einsum('nj,jkl->nkl', prior, self.cBo)
       
    def update_observation_likelihood(self):
        a = self.a[-1]
        a0 = a.sum(-1, keepdim=True)
        
        self.A = a/a0
        
        oH = - ntn(torch.sum(self.A*(psi(a+1) - psi(a0+1)), -1)) # outcome entropy

        self.oH = oH  # marginal outcome entropy
        
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
        
        fH = -ntn(torch.sum(self.D*(psi(f+1) - psi(f0+1)), -1))
        
        self.fH = torch.einsum('nc,nc->n', context, fH)
        
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
    
    def update_predictions(self, b, t):
        
        for k in range(t + 1, min(self.nt+1, t + 1 + self.depth)):
            actions = self.policies[:, k-1]
            
            beliefs = self.beliefs['offers'][b, k-1]
            predictions = torch.einsum('ni,nij->nj', beliefs, self.Bo)
            
            self.beliefs['offers'][b, k] = predictions
            outcomes = torch.einsum('ni,nijk->njk', predictions, self.A)
            
            ambiguity = torch.einsum('ni,nij->nj', predictions, self.oH)
            
            self.beliefs['outcomes'][b, k] = outcomes[:, actions]
            
            self.expectations['ambiguity'][b, t, k - 1] = ambiguity[:, actions]
            
            self.expectations['entropy'][b, t, k-1] = \
                - ntn(self.beliefs['outcomes'][b, k]*self.beliefs['outcomes'][b, k].log()).sum(-1)
        
            self.expectations['utility'][b, t, k-1] = \
                torch.einsum('nik,njk->nji', self.U, self.beliefs['outcomes'][b, k])
        
#            locations = self.beliefs['locations'][b, k-1]
#            predictions = torch.einsum('npi,ijk->npjk', locations, self.Bl)
#            self.beliefs['locations'][b, k] = predictions[:, range(len(actions)), actions]

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
            self.beliefs['points'][b, t] = 0.
            self.beliefs['points'][b, t, range(self.runs)] = torch.eye(self.nt+1)[points[:, 1:]]
        
        if t == 0:
            self.update_state_transition_matrix()
            self.F = zeros(self.runs, self.npi)
            valid = self.F == 0.
        else:
            self.F = -100 * ones(self.runs, self.npi)
            actions = self.policies[:, t-1]
            valid = locations.reshape(-1, 1) == actions.reshape(1, -1)
            valid *= self.beliefs['policies'][b, t - 1] > 0.001
            self.F[valid] = 0.        

        self.update_predictions(b, t)        
    
        U = self.expectations['utility'][b, t, t:].sum(0)
        H = self.expectations['entropy'][b, t, t:].sum(0)
        E = self.expectations['ambiguity'][b, t, t:].sum(0)
        
        G = U + H.reshape(-1, self.npi, 1) - E.reshape(-1, self.npi, 1)
        
        for i in range(3):
            G[..., i][~valid] == -100.
        
        self.expectations['EFE'][b, t] = G
        
        if t > 0:
            w = self.w[b, t-1]
            qb = 1/w
        elif b > 0:
            w = self.w[b - 1, -1]
            qb = 1/w
        else:
            qb = self.beta.clone()
            w = 1/qb
        
        w = w.reshape(-1, 1, 1)
        
        v = zeros(self.runs, 3)
        
        # posterior and prior beliefs about policies
        for i in range(self.nit):
            p = v.softmax(-1)
#            g = torch.einsum('nj,nij->ni', p, G)
            R = w*G + self.priors['policies'][-1].log().reshape(1, -1, 1) \
                + self.priors['internals'][-1].log().reshape(-1, 1, 3)
            post = (R + self.F.unsqueeze(-1)).exp()
            post /= post.sum(-1, keepdim=True).sum(-2, keepdim=True)
#            prior = R.softmax(-1)
        
#            eg = ((post - prior)*g).sum(-1)
#            dFdg = qb - self.beta + eg;
#            qb -= dFdg/4
            
#            Uv = torch.einsum('n,nij,ni->nj', w.reshape(-1), G, post) + self.priors['internals'][-1].log()
#            Fv = ntn(p*(p.log()- Uv)).sum(-1, keepdim=True)
#            v += (-v + Uv - Uv.mean(-1, keepdim=True) + Fv/3)/4
            
            #w = 1/qb.reshape(-1, 1)

        p1 = post.sum(-2)
        p2 = post.sum(-1)
        self.w[b, t] = w.reshape(-1)
        lp1 = p1.log()
        self.v[b, t] = ntn(lp1 - lp1.sum(-1, keepdim=True)/3)
        
        probs = zeros(self.runs, self.na)
        actions = self.policies[:, t]
        for a in range(self.na):
            probs[:, a] = p2[:, actions == a].sum(-1)
        
        self.beliefs['policies'][b, t] = p2
        self.logits.append(self.alpha.reshape(-1, 1)*probs.log())   
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()


