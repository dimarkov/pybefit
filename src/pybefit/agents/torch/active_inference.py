#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:14:01 2019

This module contains active inference agents for various experimental tasks. 

@author: Dimitrije Markovic
"""

import torch
from torch import zeros, ones, eye

from torch.distributions import Categorical

from numpy import nan_to_num

from itertools import product

from ..base import Discrete

__all__ = [
        'AIBandits',
]

def ntn(x):
    return torch.from_numpy(nan_to_num(x))

def psi(x):
    return torch.digamma(x)

def lngamma(x):
    return torch.lgamma(x)

class AIBandits(Discrete):
    '''Active inference agent for a multi-armed bandit problem with a hierarchical representation of the task structure.
    '''
    
    def __init__(self, pars, runs=1, blocks=1, trials=10, tm=None):
        
        self.nit = 10 # number of max iterations for belief update
        
        na = pars['na']  # number of options/arms
        ns = pars['ns']  # number of states
        self.nd = pars['nd']  # max duration length
        self.nc = pars['nc']  # number of contexts
        self.nf = pars['nf']  # number of features
        self.ni = pars['ni']  # number of internal states
        
        super().__init__(runs, blocks, trials, na, ns, self.nf)
        
        self.tm = tm

        self.priors = {}
        
        self.initiate_policies()
        self.initiate_prior_beliefs()
        self.initiate_beliefs_and_expectations()
        
        
    def initiate_beliefs_and_expectations(self):
        
        higher = {
                'context': zeros(self.nb, self.runs, self.nc, self.nd),
                'internals': zeros(self.nb, self.runs, self.ni),
                'policies': zeros(self.nb, self.runs, self.npi2)}
        
        self.beliefs = {
                'context': zeros(self.nb, self.runs, self.nc), # contexts
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states
                'outcomes': zeros(self.nb, self.nt + 1, self.runs, self.npi, self.nf),  # outcomes
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.npi, self.ns),  # selected arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf, dtype=torch.long),
                'policies': zeros(self.nb, self.nt, self.runs, self.ni, self.npi),
                'higher': higher
                }
        
        # expectations over policies
        self.expectations = {
                'higher': {'IV':[], 'EV':[]}
                }

    def set_parameters(self, x=None, epistemic=True, depth=1):
        
        self.epistemic = epistemic
        self.beta = ones(self.runs)/8
        
        self.depth = depth
        
        if x is None:
            self.a = [ones(self.runs, self.nc, self.ns, self.nf)]
        else:
            self.a = [x[-1]]
        
        self.a2 = [ones(self.runs, self.ni, self.nc, 2)]
        # forgeting rate
        self.eps = 1/5
        self.Bc = (1 - self.eps)*eye(self.nc) + self.eps*(ones(self.nc, self.nc) - eye(self.nc))/(self.nc - 1)
                
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        # variational free energy of different policies        
        self.F = zeros(self.runs, self.npi)
        
        self.logits = []  # log choice probabilities

        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
        
        self.policies2 = torch.tensor(list(product(range(self.ni), repeat=1)))
        self.npi2 = self.policies2.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('context', [])
        self.priors.setdefault('locations', [])
        self.priors.setdefault('policies', [])
        self.priors.setdefault('internals', [])

        self.priors['locations'].append(ones(self.ns)/self.ns)
        self.priors['policies'].append(ones(self.runs, self.npi)/self.npi)
        
        tm = self.tm['higher']['duration']
        self.priors['context-duration'] = torch.einsum('nc,d->ncd', 
                                                       ones(self.runs, self.nc)/self.nc,
                                                       tm[0])
        
    def update_observation_likelihood(self):
        a = self.a[-1]
        a0 = a.sum(-1, keepdim=True)
        
        self.A = a/a0
        
        oH = - ntn(torch.sum(self.A*(psi(a+1) - psi(a0+1)), -1)) # outcome entropy

        self.oH = oH  # marginal outcome entropy
        
    def update_prior_beliefs(self, b, t, **kwargs):
        
        # increase uncertainty between segments but keep expectations the same
        
        self.priors['internals'].append(self.beliefs['higher']['policies'][b])
        self.priors['context'].append(self.beliefs['higher']['context'][b].sum(-1))
        
        self.update_observation_likelihood()
        
        
    def update_beliefs_about_location(self, b, t, loc):
        
        pred = self.priors['locations'][-1]
        
        Al = torch.eye(self.ns)[loc]
        joint = Al * pred
        post = joint/joint.sum(-1, keepdim=True)
        post = post.reshape(self.runs, 1, self.ns).repeat(1, self.npi, 1)
        
        self.F = ntn(post*(joint.log().reshape(-1, 1, self.ns) - post.log())).sum(-1)
        
        self.beliefs['locations'][b, t] = post
        
    def update_higher_level_beliefs(self, b, out):
        
        pred = self.beliefs['higher']['context'][b]
        
        q_d = pred/pred.sum(-1, keepdim=True)
        
        q_c = self.beliefs['context'][b]
        post = q_d * q_c.reshape(-1, self.nc, 1)
        
        q_i = self.beliefs['internals'][b]
        
        self.beliefs['higher']['context'][b] = post
            
        a = self.a2[-1].clone()
        
        a[range(self.runs), ..., out] += torch.einsum('ni,nc->nic', q_i, q_c)
        
        self.a2.append(a)
        
    def update_beliefs(self, b, t, response_outcomes):
        
        out1, out2 = response_outcomes[-1]
        
        loc = out1['locations']
        ft = out1['features']
        
        if t == 0:
            self.outs = [(loc, ft)]
            pred_c = self.priors['context'][-1]
        else:
            self.outs.append((loc, ft))
            pred_c = self.beliefs['context'][b]
        
        A = self.A
        post_c = A[range(self.runs), :, loc, ft] * pred_c
        post_c /= post_c.sum(-1, keepdim=True)
        
        a = self.a[-1-t].clone()
        
        for out in self.outs:
            loc, ft = out
            a[range(self.runs), :, loc, ft] += post_c
        
        self.a.append(a)
        self.beliefs['context'][b] = post_c
        
        pred = self.beliefs['locations'][b, t + 1]
        
        lnA = psi(a) - psi(a.sum(-1, keepdim=True))
        
        Al = torch.eye(self.ns)[loc]
        joint = torch.einsum('ns,nps->nps', Al, pred)
        norm = joint.sum(-1)
        
        self.F += norm.log()
        
        nz = norm > 0
        joint[nz] /= norm[nz].unsqueeze(-1)
        joint[~nz] = 1/self.ns
        
        self.F += torch.einsum('ncs,nps,nc->np', lnA[range(self.runs), ..., ft], joint, post_c)
        
        self.beliefs['locations'][b, t + 1] = joint
        
        self.update_observation_likelihood()
        if t == self.nt -1:
            self.update_higher_level_beliefs(b, out2)
        
    def update_predictions(self, b, t):
        runs = self.runs
        nt = self.nt
        
        mpt = nt  # maximum number of points      
        thr = 2*mpt//3  # success threshold
        
        # predictions
        L = torch.tensor(list(product(range(self.nf), repeat= nt-t)))
        bonus = [torch.sum(L == 1, -1), torch.sum(L == 2, -1)]
        
        A = self.A
        probs = []
        if t == 0:
            context = self.priors['context'][-1]
        else:
            context = self.beliefs['context'][b]
            
        ambiguity = 0
        entropy = 0
        for k in range(t + 1, self.nt+1):
            actions = self.policies[:, k-1]
            
            self.beliefs['locations'][b, k, :, range(self.npi), actions] = 1.
            
            outcomes = torch.einsum('nc,ncpf->npf', context, A[:, :, actions])
            probs.append( outcomes[..., L[:, k - t - 1]] )
            self.beliefs['outcomes'][b, k] = outcomes
            
            ambiguity += torch.einsum('nc,ncp->np', context, self.oH[:, :, actions])
            
            entropy -= ntn(outcomes * outcomes.log()).sum(-1)
        
        probs = torch.stack(probs).prod(0)
        
        p = []
        for i in range(2):
            p.append(self.beliefs['points'][b, t][:, i + 1].reshape(-1, 1) + bonus[i].reshape(1, -1))
            
        ll = [p[0] > thr, p[1] > thr] 
        
        U = []
        lam = 1.
        for n in range(runs):
                l = ~(ll[0] + ll[1])
                l[l > 1] = 0
                
                r = 1 - probs[n, :, l[n]].sum(-1)    
                U.append(lam * (2 * r - 1))
        
        U =  torch.stack(U).reshape(-1, 1, self.npi)
        E = ambiguity.reshape(-1, 1, self.npi) 
        H = entropy.reshape(-1, 1, self.npi)
        
        if self.ni > 1:
            kappa = torch.arange(self.ni)/(self.ni-1)
            E = E * kappa.reshape(-1, 1)
            H = H * kappa.reshape(-1, 1)
        
        return U, H, E
        
        
    def plan_higher_level(self, b):
        
        lam = 2.
        U = lam*ones(self.runs, 1, 2)
        U[..., 0] = - lam
        
        policies = self.policies2
        
        actions = policies[:, 0]
        
        a = self.a2[-1]
        a0 = a.sum(-1, keepdim=True)
        
        self.A2 = a/a0
        
        A = self.A2[:, actions]
        if b > 0:
            prior = self.beliefs['higher']['context'][b-1]
        
            Bccd = self.tm['higher']['context']
            Bdd = self.tm['higher']['duration']
            
            pred = torch.einsum('dci,dj,ncd->nij', Bccd, Bdd, prior)
        
            self.beliefs['higher']['context'][b] = pred
        else:
            pred = self.priors['context-duration']
            self.beliefs['higher']['context'][b] = pred
            
        q_c = pred.sum(-1)
        marginal = torch.einsum('npco,nc->npo', A, q_c)
        
        R = (marginal * U).sum(-1)
        L = ntn(marginal * marginal.log()).sum(-1)  # risk
        H = - ntn(torch.sum(A*(psi(a[:, actions]+1) - psi(a0[:, actions]+1)), -1)) # outcome entropy
        M =  torch.einsum('npc,nc->np', H, q_c) # ambiguity
        
        self.expectations['higher']['IV'].append(R)
        self.expectations['higher']['EV'].append( -M - L)
        
        if self.epistemic:
            G = (R - L - M)
        else:
            G = R
            
        q_pi = (G/self.beta.reshape(-1, 1)).softmax(-1)
        
        self.beliefs['higher']['policies'][b] = q_pi
        
    def planning(self, b, t, offers):
        """Compute log probability of responses for the given offers. Here the offers contain various 
           information availible to the agent, like the last selected option (location) and the current 
           number of points.
        """
        if t == 0:
            self.plan_higher_level(b)
            self.update_prior_beliefs(b, t)
            locations = offers['locations']
            self.update_beliefs_about_location(b, t, locations)
        
        if 'points' in offers:
            points = offers['points']
            self.beliefs['points'][b, t] = points
        
        U, H, E = self.update_predictions(b, t)        
    
        if self.epistemic:
            G = U + H - E
        else:
            G = U
            
        valid = self.F > -10
        G = (1/self.beta).reshape(-1, 1, 1) * torch.where(valid[:, None], G, -10*torch.ones(1))
        G, _ = torch.broadcast_tensors(G, ones(self.runs, self.ni, self.npi))
        
        # posterior and prior beliefs about policies
        q_i = self.priors['internals'][-1].log()
        q_pi = self.priors['policies'][-1].log()
        for i in range(self.nit):
            R1 = torch.einsum('nip,ni->np', G, q_i) + self.priors['policies'][-1].log()
            q_pi_new = (R1 + self.F).softmax(-1)
            
            R2 = torch.einsum('nip,np->ni', G, q_pi) + self.priors['internals'][-1].log()
            q_i_new = R2.softmax(-1)
            
            q_i = q_i_new.clone()
            q_pi = q_pi_new.clone()
    
        probs = zeros(self.runs, self.na)
        actions = self.policies[:, t]

        for a in range(self.na):
            probs[:, a] = q_pi[:, actions == a].sum(-1)
        
        self.beliefs['policies'][b, t] = q_pi.reshape(-1, 1, self.npi)
        self.beliefs['internals'][b] = q_i
        self.logits.append(100 * probs.log())
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()

