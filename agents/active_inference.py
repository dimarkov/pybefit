#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:14:01 2019

This module contains active inference agents for various experimental tasks. 

@author: Dimitrije Markovic
"""

import torch
from torch import zeros, ones, tensor, eye

from torch.distributions import Categorical

from numpy import nan_to_num

from itertools import product

from .agent import Discrete

__all__ = [
        'AIBandits',
        'AIBanditsNaive',
        'AIBanditsFlat',
        'AIBanditsDeep'
]

def ntn(x):
    return torch.from_numpy(nan_to_num(x))

def psi(x):
    return torch.digamma(x)

def lngamma(x):
    return torch.lgamma(x)

class AIBandits(Discrete):
    '''Agent using backward induction to compute optimal choice.
    '''
    def __init__(self, pars, runs=1, blocks=1, trials=10, link=None):
        
        self.nit = 10
        
        na = pars['na']  # number of choices
        ns = pars['ns']  # number of states
        no = pars['no']  # number of outcomes
        self.nf = pars['nf']  # number of factors
        self.ni = pars['ni']  # number of internal states
        self.link = link
        
        super(AIBandits, self).__init__(runs, blocks, trials, na, ns, no)
                
        self.priors = {}
        
        self.initiate_policies()
        self.initiate_beliefs_and_expectations()
        
    def initiate_beliefs_and_expectations(self):
        
        self.beliefs = { 
                'internals': [],  # internal states
                'policies': [],  # policies
                'externals': [],  # external states
                }
        
        self.predictions = {
                'externals': [],  # external states
                'outcomes': []  # outcomes
                }
        
        # expectations over policies
        self.expectations = {
                'utility': [],
                'ambiguity': [],
                'entropy': [],
                'EFE': []
                }


    def set_parameters(self, 
                       x=None, 
                       priors=None, 
                       epistemic=True,
                       depth=1,
                       lam=.1):
        
        self.epistemic = epistemic
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)
        
        if x is None:
            self.a = [ones(self.runs, self.no, self.ns, self.nf)]
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
        
        self.w = zeros(self.nb, self.nt, self.runs)
        self.v = zeros(self.nb, self.nt, self.runs, 3)
        
        self.logits = []
        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.size['policies'] = self.policies.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        # prior beliefs about external states
        self.priors.setdefault('what', [])
        self.priors.setdefault('where', [])
        self.priors.setdefault('when', [])
        
        # prior beliefs about internal states
        self.priors.setdefault('internals', [])
        
        # prior beliefs about policies
        self.priors.setdefault('policies', [])
        
        keys = ['what', 'where', 'when', 'internals', 'policies']
        for key in keys:
            if key in kwargs:
                self.priors[key].append(kwargs[key])
            else:
                n = self.size[key]
                self.priors[key].append(ones(self.runs, n)/n)
    
    def update_state_transition_matrix(self):
        prior = self.priors['context'][-1]
        self.Bo = torch.einsum('nj,jkl->nkl', prior, self.cBo)
       
    def update_observation_likelihood(self):
        a = self.a[-1]
        a0 = self.a[-1].sum(-1, keepdim=True)
        
        self.A = a/a0
        
        oH = - ntn(torch.sum(self.A*(psi(a+1) - psi(a0+1)), -1))  # outcome entropy

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
                
    def update_higher_level(self, b, t, out1, out2):
        
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
        out = response_outcomes[-1]
        
        offers = self.beliefs['offers'][b, t]
        
        A = self.A[range(self.runs), ..., out[0]]
        post = torch.einsum('ni,nij->nij', offers, A)[range(self.runs), :, res]
        post /= post.sum(-1, keepdim=True)
        
        self.beliefs['offers'][b, t] = post
        self.beliefs['offers'][b, t + 1] = torch.einsum('nij,ni->nj', self.Bo, self.beliefs['offers'][b, t])
        
        self.beliefs['locations'][b, t] = torch.eye(self.na)[res]
        
        self.update_second_level(b, t, out1, out2)
    
    def update_beliefs_and_expectations(self, b, t):
        runs = self.runs
        nt = self.nt
        
        mpt = nt  # maximum number of points      
        thr = 2*mpt//3  # success threshold
        
        # update beliefs
        
        log_where = zeros(self.nt, self.runs, self.npi, self.nl)
        log_what = zeros(self.nt, self.runs, self.npi, self.nz)
        log_when = zeros(self.nt, self.runs, self.npi, self.nd)
        
        # predictions
        L = torch.tensor(list(product(range(self.nf), repeat= nt-t)))
        bonus = [torch.sum(L == 1, -1), torch.sum(L == 2, -1)]
        
        actions = torch.arange(self.na)
        log_probs = 0.
        for k in range(t+1, nt+1):
            actions = self.policies[:, k-1]
            
            beliefs = self.beliefs['offers'][b, k-1]
            predictions = torch.einsum('ni,nij->nj', beliefs, self.Bo)
            self.beliefs['offers'][b, k] = predictions
            
            ambiguity = torch.einsum('ni,nij->nj', predictions, self.oH)

            outcomes = torch.einsum('ni,nijk->njk', predictions, self.A)[:, actions]
            log_probs += outcomes[..., L[:, k - t - 1]].log()
            
            self.expectations['ambiguity'][b, t, k - 1] = ambiguity[:, actions]
            
            self.expectations['entropy'][b, t, k-1] = - ntn(outcomes * outcomes.log()).sum(-1)
        
        probs = log_probs.exp()
        
        p = []
        for i in range(2):
            p.append(self.beliefs['points'][b, t][:, i + 1].reshape(-1, 1) + bonus[i].reshape(1, -1))
            
        ll = [p[0] > thr, p[1] > thr] 
        
        U = []
        for n in range(runs):
                l = ~(ll[0] + ll[1])
                l[l > 1] = 0
                
                r1 = probs[n, :, ll[0][n]].sum(-1)
                r2 = probs[n, :, ll[1][n]].sum(-1)
                r = 1 - probs[n, :, l[n]].sum(-1)
                U.append(torch.stack([2*r - 1, 2*r1 - 1, 2*r2 - 1], -1))
        
        self.expectations['utility'][b, t, -1] = U
        
    def plan_higher_level(self, b, t, offers):
        
        agent = self.link['agent']
        agent.planning(b, t, offers)
        beliefs = agent.beliefs['externals']['what'][-1]
        
        if 'a' in self.link:
            a = self.link['a']
            bara = torch.einsum('nsqo,ns->nqo', a, beliefs)
            self.a.append(bara)
        
        if 'u' in self.link: 
            u = self.link['u']
            baru = torch.einsum('nso,ns->no', u, beliefs)
            self.U.append(baru)
        
        if 'b' in self.link:
            b = self.link['b']
            barb = torch.einsum('nspqq,ns->npqq', b, beliefs)
            self.b.append(barb)
        
    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
        """
        if t == 0:
            self.plan_higher_level(b, t, offers)
        
        if 'where' in offers:
            where = offers['where']
            self.beliefs['externals']['where'].append(where)
        if 'what' in offers:
            what = offers['what']
            self.beliefs['externals']['what'].append(what)
        if 'when' in offers:
            when = offers['when']
            self.beliefs['externals']['when'].append(when)
            
        self.update_beliefs_and_expecations(b, t)
        
        if self.epistemic:
            U = self.expectaions['utility'][-1]
            H = self.expectations['entropy'][-1]
            E = self.expectations['ambiguity'][-1]
            G = U + H.reshape(-1, self.npi, 1) - E.reshape(-1, self.npi, 1)
        else:
            G = self.expectations['utility'][-1]
        
        w = self.w[-1]
        qb = 1/w
        w = w.reshape(-1, 1, 1)
        
        # posterior and prior beliefs about policies
        for i in range(self.nit):
            J = w*G + self.priors['policies'][-1].log().reshape(-1, self.npi, 1) \
                + self.priors['internals'][-1].log().reshape(-1, 1, self.ni)
            post = (J + self.F.unsqueeze(-1)).exp()
            post /= post.sum(-1, keepdim=True).sum(-2, keepdim=True)

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

class AIBanditsNaive(Discrete):
    '''Active inference agent for a multi-armed bandit problem with flat representation of the task structure.
    '''
    
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        self.nit = 1
        
        na = pars['na']  # number of choices/arms
        ns = pars['ns']  # number of states
        self.nf = pars['nf']  # number of features
        self.ni = pars['ni']  # number of internal states
        
        super(AIBanditsNaive, self).__init__(runs, blocks, trials, na, ns, [])
                
        self.priors = {}
        
        self.initiate_policies()
        self.initiate_prior_beliefs()
        self.initiate_beliefs_and_expectations()
        
    def initiate_beliefs_and_expectations(self):
        
        self.beliefs = { 
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states
                'outcomes': zeros(self.nb, self.nt + 1, self.runs, self.npi, self.nf),  # outcomes
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.npi, self.ns),  # selected arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf, dtype=torch.long),
                'policies': zeros(self.nb, self.nt, self.runs, self.ni, self.npi)
                }
        
        # expectations over policies
        self.expectations = {
                'utility': zeros(self.nb, self.nt, self.runs, self.ni, self.npi),
                'ambiguity': zeros(self.nb, self.nt, self.runs, self.npi),
                'entropy': zeros(self.nb, self.nt, self.runs, self.npi),
                'EFE': zeros(self.nb, self.nt, self.runs, self.npi)
                }

    def set_parameters(self, x=None, epistemic=True, depth=1, lam=1.):
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)/8
        
        self.depth = depth
        
        if x is None:
            self.a = [ones(self.runs, self.na, self.nf)]
        else:
            self.a = [x[-1]]
        
        self.i = [ones(self.runs, self.ni)]
        
        # forgeting rate
        self.eps = 1/5
        
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        self.lam = lam
        
        self.epistemic = epistemic
        
        self.w = zeros(self.nb, self.nt, self.runs)
        
        self.F = zeros(self.runs, self.npi)
        
        self.logits = []  # log choice probabilities are stored here

        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('locations', [])
        self.priors.setdefault('policies', [])
        self.priors.setdefault('internals', [])

        self.priors['locations'].append(ones(self.ns)/self.ns)
        self.priors['policies'].append(ones(self.npi)/self.npi)
        
    def update_observation_likelihood(self):
        a = self.a[-1]
        a0 = a.sum(-1, keepdim=True)
        
        self.A = a/a0
        
        oH = - ntn(torch.sum(self.A*(psi(a+1) - psi(a0+1)), -1)) # outcome entropy

        self.oH = oH  # marginal outcome entropy
    
    def update_prior_beliefs(self, b, t, **kwargs):
        
        a = self.a[-1]
        a0 = a.sum(-1, keepdim=True)
        
        A = a/a0
        
        # increase uncertainty between segments but keep expectations the same
        if b > 0:
            q_i = self.beliefs['internals'][b-1]
            i = self.i[-1] + q_i
            i0 = i.sum(-1, keepdim = True)
            
            I = i/i0            
            
            v1 = a0*(1 - self.eps) + self.eps
            self.a.append(v1*A)
            
            v2 = i0 * (1 - self.eps) + self.eps
            self.i.append(v2*I)
        else:
            i = self.i[-1]
            i0 = i.sum(-1, keepdim = True)
            
            I = i/i0

        self.priors['internals'].append(I)
        self.update_observation_likelihood()
        
        
    def update_beliefs_about_location(self, b, t, loc):
        
        pred = self.priors['locations'][-1]
        
        Al = torch.eye(self.ns)[loc]
        joint = Al * pred
        post = joint/joint.sum(-1, keepdim=True)
        post = post.reshape(self.runs, 1, self.ns).repeat(1, self.npi, 1)
        
        self.F = ntn(post*(joint.log().reshape(-1, 1, self.ns) - post.log())).sum(-1)
        
        self.beliefs['locations'][b, t] = post
        
    def update_beliefs(self, b, t, response_outcomes):
        
        out1, out2 = response_outcomes[-1]
        
        loc = out1['locations']
        ft = out1['features']
        
        a = self.a[-1].clone()
        
        a[range(self.runs), loc, ft] += 1.
        
        self.a.append(a)
        
        pred = self.beliefs['locations'][b, t + 1]
        
        lnA = psi(a) - psi(a.sum(-1, keepdim=True))
        
        Al = torch.eye(self.ns)[loc]
        joint = torch.einsum('ns,nps->nps', Al, pred)
        norm = joint.sum(-1)
        
        self.F += norm.log()
        
        nz = norm > 0
        joint[nz] /= norm[nz].unsqueeze(-1)
        joint[~nz] = 1/self.ns
        
        self.F += torch.einsum('ns,nps->np', lnA[range(self.runs), :, ft], joint)
        
        self.beliefs['locations'][b, t + 1] = joint
        
        self.update_observation_likelihood()
        
    def update_predictions(self, b, t):
        runs = self.runs
        nt = self.nt
        
        mpt = nt  # maximum number of points      
        thr = 2*mpt//3  # success threshold
        
        # predictions
        L = torch.tensor(list(product(range(self.nf), repeat= nt-t)))
        bonus = [torch.sum(L == 1, -1), torch.sum(L == 2, -1)]
        
        policies = self.policies
        npi = len(policies)
        
        A = self.A
        probs = []
        for k in range(t + 1, self.nt+1):
            actions = policies[:, k-1]
            
            self.beliefs['locations'][b, k, :, range(npi), actions] = 1.
            
            outcomes = A[:, actions]
            probs.append( outcomes[..., L[:, k - t - 1]] )
            self.beliefs['outcomes'][b, k] = outcomes
            
            self.expectations['ambiguity'][b, t] += self.oH[:, actions]
            
            self.expectations['entropy'][b, t] -= ntn(outcomes * outcomes.log()).sum(-1)
        
        probs = torch.stack(probs).prod(0)
        
        p = []
        for i in range(2):
            p.append(self.beliefs['points'][b, t][:, i + 1].reshape(-1, 1) + bonus[i].reshape(1, -1))
            
        ll = [p[0] > thr, p[1] > thr] 
        
        U = []
        for n in range(runs):
                l = ~(ll[0] + ll[1])
                
                r = 1 - probs[n, :, l[n]].sum(-1)    
                U.append(2 * r - 1)

#                r1 = probs[n, :, ll[0][n]].sum(-1)
#                r2 = probs[n, :, ll[1][n]].sum(-1)
        
        lam = self.lam        
        self.expectations['utility'][b, t] = torch.stack(U).reshape(-1, 1, self.npi) * lam.reshape(1, -1, 1)
        
#        self.priors['internals'].append(self.expectations['utility'][b, t].sum(0).softmax(-1))

    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if t == 0:
            self.update_prior_beliefs(b, t)
            locations = offers['locations']
            self.update_beliefs_about_location(b, t, locations)
        
        if 'points' in offers:
            points = offers['points']
            self.beliefs['points'][b, t] = points
#            self.beliefs['points'][b, t] = 0.
#            self.beliefs['points'][b, t, range(self.runs)] = torch.eye(self.nt+1)[points[:, 1:]]
        
        self.update_predictions(b, t)        
    
        U = self.expectations['utility'][b, t]
        H = self.expectations['entropy'][b, t]
        E = self.expectations['ambiguity'][b, t]
        
        invalid = self.F <= -10
        
        if self.epistemic:
            G = U + H.reshape(-1, 1, self.npi) - E.reshape(-1, 1, self.npi)
        else:
            G = U
        for i in range(self.ni):
             G[:, i][invalid] == -10.
        
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
        
#        v = zeros(self.runs, 3)
        
        # posterior and prior beliefs about policies
        for i in range(self.nit):
#            p = v.softmax(-1)
#            g = torch.einsum('nj,nij->ni', p, G)
            R = w*G \
                + self.priors['internals'][-1].log().reshape(-1, self.ni, 1)
            
            q_ipi = (R + self.F.reshape(-1, 1, self.npi)).reshape(self.runs, -1).softmax(-1)
            
#            q_i = ntn(q_pii*(R - q_pii.log())).sum(-2).softmax(-1)
#            prior = R.softmax(-1)
        
#            eg = ((post - prior)*g).sum(-1)
#            dFdg = qb - self.beta + eg;
#            qb -= dFdg/4
            
#            Uv = torch.einsum('n,nij,ni->nj', w.reshape(-1), G, post) + self.priors['internals'][-1].log()
#            Fv = ntn(p*(p.log()- Uv)).sum(-1, keepdim=True)
#            v += (-v + Uv - Uv.mean(-1, keepdim=True) + Fv/3)/4
            
            #w = 1/qb.reshape(-1, 1)
            
        self.w[b, t] = w.reshape(-1)
#        lp1 = q_i.log()
#        self.v[b, t] = ntn(lp1 - lp1.sum(-1, keepdim=True)/self.ni)
        
#        q_pi = torch.einsum('npi,ni->np', q_pii, q_i)        
        probs = zeros(self.runs, self.na)
        actions = self.policies[:, t]
        q_ipi = q_ipi.reshape(-1, self.ni, self.npi)
        q_i = q_ipi.sum(-1, keepdim=True)
        q_pi = q_ipi.sum(-2)

        for a in range(self.na):
            probs[:, a] = q_pi[:, actions == a].sum(-1)
        
        self.beliefs['policies'][b, t] = q_ipi/q_i
        self.beliefs['internals'][b] = q_i[..., 0]
        self.logits.append(self.alpha.reshape(-1, 1)*probs.log())   
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()
    
class AIBanditsFlat(Discrete):
    '''Active inference agent for a multi-armed bandit problem with flat representation of the task structure.
    '''
    
    def __init__(self, pars, runs=1, blocks=1, trials=10, tm=None):
        
        self.nit = 1
        
        na = pars['na']  # number of choices/arms
        ns = pars['ns']  # number of states
        self.nd = pars['nd']
        self.nc = pars['nc']  # number of contexts
        self.nf = pars['nf']  # number of features
        self.ni = pars['ni']  # number of internal states
        
        super(AIBanditsFlat, self).__init__(runs, blocks, trials, na, ns, [])
        
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
                'utility': zeros(self.nb, self.nt, self.runs, self.ni, self.npi),
                'ambiguity': zeros(self.nb, self.nt, self.runs, self.ni, self.npi),
                'entropy': zeros(self.nb, self.nt, self.runs, self.ni, self.npi),
                'EFE': zeros(self.nb, self.nt, self.runs, self.npi),
                'higher': {'IV':[], 'EV':[]}
                }

    def set_parameters(self, x=None, epistemic=True, depth=1, lam=1.):
        
        self.alpha = 100*ones(self.runs)
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
        
        self.lam = lam
        
        self.epistemic = epistemic
        
        self.w = zeros(self.nb, self.nt, self.runs)
        
        self.F = zeros(self.runs, self.npi)
        
        self.logits = []  # log choice probabilities are stored here

        
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
        self.priors['policies'].append(ones(self.npi)/self.npi)
        
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
        for n in range(runs):
                l = ~(ll[0] + ll[1])
                l[l > 1] = 0
                
                r = 1 - probs[n, :, l[n]].sum(-1)    
                U.append(2 * r - 1)

                
        lam = self.lam        
        self.expectations['utility'][b, t] = torch.stack(U).reshape(-1, 1, self.npi)
        self.expectations['ambiguity'][b, t] = ambiguity.reshape(-1, 1, self.npi) * lam.reshape(1, -1, 1)
        self.expectations['entropy'][b, t] = entropy.reshape(-1, 1, self.npi) * lam.reshape(1, -1, 1)
        
        
    def plan_higher_level(self, b):
        
        lam = 1.
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
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if t == 0:
            self.plan_higher_level(b)
            self.update_prior_beliefs(b, t)
            locations = offers['locations']
            self.update_beliefs_about_location(b, t, locations)
        
        if 'points' in offers:
            points = offers['points']
            self.beliefs['points'][b, t] = points
#            self.beliefs['points'][b, t] = 0.
#            self.beliefs['points'][b, t, range(self.runs)] = torch.eye(self.nt+1)[points[:, 1:]]
        
        self.update_predictions(b, t)        
    
        U = self.expectations['utility'][b, t]
        H = self.expectations['entropy'][b, t]
        E = self.expectations['ambiguity'][b, t]
        
        valid = self.F > -10
        
        if self.epistemic:
            G = U + H - E
        else:
            G = U
        for i in range(self.ni):
             G[:, i][~valid] == -10.
        
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
        
#        v = zeros(self.runs, 3)
        
        # posterior and prior beliefs about policies
        for i in range(self.nit):
#            p = v.softmax(-1)
#            g = torch.einsum('nj,nij->ni', p, G)
            R = w*G + self.priors['policies'][-1].log() + self.priors['internals'][-1].log().reshape(-1, self.ni, 1)
            V = (R + self.F.reshape(-1, 1, self.npi)).reshape(self.runs, -1)
            q_ipi = V.softmax(-1)

#            q_i = ntn(q_pii*(R - q_pii.log())).sum(-2).softmax(-1)
#            prior = R.softmax(-1)
        
#            eg = ((post - prior)*g).sum(-1)
#            dFdg = qb - self.beta + eg;
#            qb -= dFdg/4
            
#            Uv = torch.einsum('n,nij,ni->nj', w.reshape(-1), G, post) + self.priors['internals'][-1].log()
#            Fv = ntn(p*(p.log()- Uv)).sum(-1, keepdim=True)
#            v += (-v + Uv - Uv.mean(-1, keepdim=True) + Fv/3)/4
            
            #w = 1/qb.reshape(-1, 1)
            
        self.w[b, t] = w.reshape(-1)
#        lp1 = q_i.log()
#        self.v[b, t] = ntn(lp1 - lp1.sum(-1, keepdim=True)/self.ni)
        
#        q_pi = torch.einsum('npi,ni->np', q_pii, q_i)        
        probs = zeros(self.runs, self.na)
        actions = self.policies[:, t]
        q_ipi = q_ipi.reshape(-1, self.ni, self.npi)
        q_i = q_ipi.sum(-1, keepdim=True)
        q_pi = q_ipi.sum(-2)

        for a in range(self.na):
            probs[:, a] = q_pi[:, actions == a].sum(-1)
        
        self.beliefs['policies'][b, t] = q_ipi/q_i
        self.beliefs['internals'][b] = q_i[..., 0]
        self.logits.append(self.alpha.reshape(-1, 1)*probs.log())   
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()

class AIBanditsDeep(Discrete):
    
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        self.nit = 1
        
        na = pars['na']  # number of choices/arms
        no = pars['no']  # number of offers
        self.nf = pars['nf']  # number of features
        self.nc = pars['nc']  # number of contexts
        self.ni = pars['ni']  # number of internal states
        
        super(AIBanditsDeep, self).__init__(runs, blocks, trials, na, None, no)
                
        self.priors = {}
        
        self.initiate_policies()
        self.initiate_prior_beliefs()
        self.initiate_beliefs_and_expectations()
        
    def initiate_beliefs_and_expectations(self):
        
        self.beliefs = { 
                'context': zeros(self.nb, self.runs, self.nc),  # context states 2nd level
                'internals': zeros(self.nb, self.runs, self.ni),  # internal states 2nd level
                'outcomes2': None,  # outcomes at the second level
#                'offers': zeros(self.nb, self.nt + 1, self.runs, self.no),  # offer state first level
                'outcomes': zeros(self.nb, self.nt + 1, self.runs, self.npi, self.nf),  # outcomes first level
                'locations': zeros(self.nb, self.nt + 1, self.runs, self.na),  # selected arm number
                'points': zeros(self.nb, self.nt + 1, self.runs, self.nf, dtype=torch.long),
                'policies': zeros(self.nb, self.nt, self.runs, self.npi)
                }
        
        # expectations over policies
        self.expectations = {
                'utility': zeros(self.nb, self.nt, self.runs, self.npi),
                'ambiguity': zeros(self.nb, self.nt, self.runs, self.npi),
                'entropy': zeros(self.nb, self.nt, self.runs, self.npi),
                'EFE': zeros(self.nb, self.nt, self.runs, self.npi)
                }

    def set_parameters(self, x=None, normalise=True, depth=1, lam=1.):
        
        self.alpha = 100*ones(self.runs)
        self.beta = ones(self.runs)/10
        
        self.depth = depth
        
        if x is None:
            self.a = {'local': [ones(self.runs, self.na, self.nf)],
                      'link' : [ones(self.runs, self.nc, self.na, self.nf)]}
        else:
            self.a = x[-1]
        
        rho2 = 1 - 1/20
        M = torch.diag(ones(self.nc-1), 1) + torch.diag(ones(self.nc-2), -2)
        self.Bc = rho2*eye(self.nc) + (1-rho2)*M 
        
        # choice dependent location transitions
        self.Bl = eye(self.na).repeat(self.na, 1, 1)
        
        self.lam = lam
        
        self.w = zeros(self.nb, self.nt, self.runs)
        self.v = zeros(self.nb, self.nt, self.runs, 3)
        
        self.logits = []  # log choice probabilities are stored here
        
    def initiate_policies(self):
        self.policies = torch.tensor(list(product(range(self.na), repeat=self.nt)))
        self.npi = self.policies.shape[0]
    
    def initiate_prior_beliefs(self, **kwargs):
        
        self.priors.setdefault('context', [])
        self.priors.setdefault('locations', [])
        self.priors.setdefault('policies', [])
        self.priors.setdefault('internals', [])

        if 'context' in kwargs:
            self.priors['context'].append(kwargs['context'])
        else:
            self.priors['context'].append(torch.tensor([1., 0., 0., 0., 0., 0.]).repeat(self.runs, 1))
        
        self.priors['locations'].append(ones(self.na)/self.na)
        self.priors['policies'].append(ones(self.npi)/self.npi)
        
#        self.priors['internals'].append(torch.tensor([1 - .999999, .999999, 1 - .999999]).repeat(self.runs, 1))
        
    def update_observation_likelihood(self):
        a = self.a['local'][-1]
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
        
        a = self.a['link'][-1]
        
        # set prior over offers as a marginal over contexts
        self.a['local'].append(torch.einsum('nc,ncaf->naf', context, a))
        
        self.update_observation_likelihood()
        
    def update_second_level(self, b, t, out1, out2):
        
#        context = self.priors['context'][-1]
#        locations = self.beliefs['locations'][b, t]
        
#        alpha = self.beliefs['offers'][b, :t+1]
#        gamma = self.beliefs['offers'][b, :t+2].clone()
#        tm = self.Bo
#        for k in range(t+1):
#            pred = torch.einsum('ni,nij->nj', alpha[-k-1], tm)
#            gamma[-k-2] = torch.einsum('nj,nij,ni->ni', gamma[-k-1]/pred, tm, alpha[-k-1])
        
        cp = self.priors['context'][-1]
#        op = self.priors['offers'][-1]/self.priors['offers'][-1].sum(-1, keepdim=True)
#        context = (torch.einsum('nco,no->nc', op.log(), gamma[0]) + cp.log()).softmax(-1)
        
        a1 = self.a['local'][-1]
        a10 = a1.sum(-1, keepdim=True)
        a2 = self.a['link'][-1]
        a20 = a2.sum(-1, keepdim=True)
        
        exp1 = (a2 - 1)*(psi(a1) - psi(a10))[:, None] - lngamma(a2).sum(-1, keepdim=True) + lngamma(a20)
        
        context = (cp.log() + exp1.reshape(self.runs, self.nc, -1).sum(-1)).softmax(-1)
        
#        f = torch.einsum('nc,nf->ncf', context, gamma[0]) + self.priors['offers'][-1]  
#        a = torch.einsum('no,nl,nf->nolf', offers, locations, out1) + self.a[-1]
        
#        self.a.append(a)
#        self.priors['offers'].append(f)
#        self.update_observation_likelihood()
        
        if t == self.nt-1:
            self.a['link'].append(a2 + torch.einsum('nc,naf->ncaf', context, a1))
            
            self.priors['context'].append(context@self.Bc)
            
    def update_beliefs(self, b, t, response_outcomes):
        
        res = response_outcomes[-2]
        out1, out2 = response_outcomes[-1]
        
        a = self.a['local'][-1].clone()
        
        a[range(self.runs), res] += out1
        
        self.a['local'].append(a)
        
        self.beliefs['locations'][b, t] = torch.eye(self.na)[res]
        
        self.update_observation_likelihood()
        
        if t == self.nt-1:
            self.update_second_level(b, t, out1, out2)
        
#    def update_beliefs(self, b, t, response_outcomes):
#        
#        res = response_outcomes[-2]
#        out1, out2 = response_outcomes[-1]
#        
#        offers = self.beliefs['offers'][b, t]
#        
#        A = self.A[range(self.runs), :, :, out1.argmax(-1)]
#        post = torch.einsum('ni,nij->nij', offers, A)[range(self.runs), :, res]
#        post /= post.sum(-1, keepdim=True)
#        
#        self.beliefs['offers'][b, t] = post
#        self.beliefs['offers'][b, t + 1] = torch.einsum('nij,ni->nj', self.Bo, self.beliefs['offers'][b, t])
#        
#        self.beliefs['locations'][b, t] = torch.eye(self.na)[res]
#        
#        self.update_second_level(b, t, out1, out2)
        
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
        for k in range(t + 1, self.nt+1):
            actions = self.policies[:, k-1]
            
            outcomes = A[:, actions]
            probs.append( outcomes[..., L[:, k - t - 1]] )
            self.beliefs['outcomes'][b, k] = outcomes
            
            self.expectations['ambiguity'][b, t] += self.oH[:, actions]
            
            self.expectations['entropy'][b, t] -= ntn( outcomes * outcomes.log() ).sum(-1)
        
        probs = torch.stack(probs).prod(0)
        
        p = []
        for i in range(2):
            p.append(self.beliefs['points'][b, t][:, i + 1].reshape(-1, 1) + bonus[i].reshape(1, -1))
            
        ll = [p[0] > thr, p[1] > thr] 
        
        U = []
        lam = self.lam
        for n in range(runs):
                l = ~(ll[0] + ll[1])
                l[l > 1] = 0
                
                r = 1 - probs[n, :, l[n]].sum(-1)    
                U.append(lam*(2 * r - 1))

#                r1 = probs[n, :, ll[0][n]].sum(-1)
#                r2 = probs[n, :, ll[1][n]].sum(-1)
                
        self.expectations['utility'][b, t] = torch.stack(U)
    
    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if 'context' in offers:
            self.update_prior_beliefs(b, t, **offers)
        elif t == 0:
            self.update_prior_beliefs(b, t)
        
        if 'locations' in offers:
            locations = offers['locations']
            self.beliefs['locations'][b, t] = 0.
            self.beliefs['locations'][b, t, range(self.runs), locations] = 1.
        
        if 'points' in offers:
            points = offers['points']
            self.beliefs['points'][b, t] = points
#            self.beliefs['points'][b, t] = 0.
#            self.beliefs['points'][b, t, range(self.runs)] = torch.eye(self.nt+1)[points[:, 1:]]
        
        if t == 0:
#            self.update_state_transition_matrix()
            self.F = zeros(self.runs, self.npi)
            valid = self.F == 0.
        else:
            self.F = -10 * ones(self.runs, self.npi)
            actions = self.policies[:, t-1]
            valid = locations.reshape(-1, 1) == actions.reshape(1, -1)
            valid *= self.beliefs['policies'][b, t - 1] > 0.001
            self.F[valid] = 0.        

        self.update_predictions(b, t)        
    
        U = self.expectations['utility'][b, t]
        H = self.expectations['entropy'][b, t]
        E = self.expectations['ambiguity'][b, t]
        
        G = U + H - E
        G[~valid] = -10.
        
#        for i in range(3):
#            G[..., i][~valid] == -10.
        
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
        
        w = w.reshape(-1, 1)
        
#        v = zeros(self.runs, 3)
        
        # posterior and prior beliefs about policies
        for i in range(self.nit):
#            p = v.softmax(-1)
#            g = torch.einsum('nj,nij->ni', p, G)
            R = w*G + self.priors['policies'][-1].log()
            q_pi = (R + self.F).softmax(-1)

#            q_i = ntn(q_pii*(R - q_pii.log())).sum(-2).softmax(-1)
#            prior = R.softmax(-1)
        
#            eg = ((post - prior)*g).sum(-1)
#            dFdg = qb - self.beta + eg;
#            qb -= dFdg/4
            
#            Uv = torch.einsum('n,nij,ni->nj', w.reshape(-1), G, post) + self.priors['internals'][-1].log()
#            Fv = ntn(p*(p.log()- Uv)).sum(-1, keepdim=True)
#            v += (-v + Uv - Uv.mean(-1, keepdim=True) + Fv/3)/4
            
            #w = 1/qb.reshape(-1, 1)
            
        self.w[b, t] = w.reshape(-1)
#        lp1 = q_i.log()
#        self.v[b, t] = ntn(lp1 - lp1.sum(-1, keepdim=True)/self.ni)
        
#        q_pi = torch.einsum('npi,ni->np', q_pii, q_i)        
        probs = zeros(self.runs, self.na)
        actions = self.policies[:, t]
        for a in range(self.na):
            probs[:, a] = q_pi[:, actions == a].sum(-1)
        
        self.beliefs['policies'][b, t] = q_pi
        self.logits.append(self.alpha.reshape(-1, 1)*probs.log())  
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()


