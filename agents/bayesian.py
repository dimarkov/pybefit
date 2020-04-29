#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains Bayesian agents for various experimental tasks
Created on Mon Jan 21 13:50:30 2019

@author: Dimitrije Markovic
"""

import warnings
import torch
from torch import zeros, ones, tensor

from torch.distributions import Bernoulli, Categorical

from .agent import Discrete

softplus = torch.nn.functional.softplus

__all__ = [
        'HGFSocInf',
        'SGFSocInf',
        'BayesTempRevLearn'
]

class HGFSocInf(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=120):
        
        na = 2  # number of choices
        ns = 2  # number of states
        no = 2  # number of outcomes
        
        super(HGFSocInf, self).__init__(runs, blocks, trials, na, ns, no)
        
    def set_parameters(self, x=None):
        
        
        if x is not None:
            self.mu0 = torch.zeros_like(x[..., :2])
            self.mu0[..., 0] = 0.
            self.mu0[..., 1] = -(x[..., 0]+5.).relu() - 2.
            self.pi0 = torch.zeros_like(x[..., 2:4])
            self.pi0[..., 0] = 2.
            self.pi0[..., 1] = 2.
            self.eta = (x[..., 1]-5).sigmoid()
            self.zeta = x[..., 2].sigmoid()
            self.beta = x[..., 3].exp()
            self.bias = x[..., 4]
        else:
            self.mu0 = zeros(self.runs, 2)
            self.pi0 = ones(self.runs, 2) 
            self.eta = .1*ones(self.runs)
            self.zeta = .95*ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.bias = zeros(self.runs)

        self.kappa = .5
        
        #set initial beliefs
        self.mu = [self.mu0]
        self.pi = [self.pi0]
        
        self.npar = 5       
        self.offers = []
        self.logits = []

    def update_beliefs(self, b, t, response_outcomes, mask=None):
        
        if mask is None:
            mask = ones(self.runs)
        
        mu = [] 
        pi = []
        
        mu_pre = self.mu[-1]
        pi_pre = self.pi[-1]
        
        # 1st level
        # Prediction
        muhat = mu_pre[:, 0].sigmoid()

        #Update

        # 2nd level
        # Precision of prediction
        
        w1 = torch.exp(self.kappa*mu_pre[:, -1])

        # Updates
        pihat1 = pi_pre[:, 0]/(1 + pi_pre[:, 0]*w1)

        pi1 = pihat1 + mask * muhat * (1-muhat)
        
        o = response_outcomes[-1][:, -2]  # observation       
        wda = mask * ( (o + 1)/2 - muhat) / pi1
        mu.append(mu_pre[:, 0] + wda)
        pi.append(pi1)


        # Volatility prediction error
        da1 = mask * ((1 / pi1 + (wda)**2) * pihat1 - 1)

        # 3rd level
        # Precision of prediction
        pihat2 = pi_pre[:, 1] / (1. + pi_pre[:, 1] * self.eta)

        # Weighting factor
        w2 = w1 * pihat1 * mask

        # Updates
        pi2 = pihat2 + self.kappa**2 * w2 * (w2 + (2 * w2 - 1) * da1) / 2

        mu.append(mu_pre[:, 1] + self.kappa * w2 * da1 / (2 * pi2))
        pi.append(pi2)
        
        invalid = (pi[-1] <= 0.) | torch.isnan(mu[0]) | torch.isnan(mu[1])
        
        mu = torch.stack(mu, dim=-1)
        pi = torch.stack(pi, dim=-1)
                
        if invalid.any():
            # negative precision is impossible, hence parameter values are unreliable. 
            # Normaly one can rise an error but we can also just fix the values to 
            # zero and prevent the update at all levels. One would expect that 
            	# fixed values provide bad fit to the data.
            warnings.warn('Encountered negative precision on the 3rd level')
            pi[invalid] = 0.
            mu[invalid] = 0.
            
        self.mu.append(mu)
        self.pi.append(pi)

    def planning(self, b, t, offers):
        
        sig = (self.kappa*self.mu[-1][:, 1]).exp() + self.pi[-1][:, 0]
        gamma = torch.sqrt(1 + 3.14159*sig/8)
        b_soc = (self.mu[-1][:, 0]/gamma).sigmoid()
        
        b_vis = offers

        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log()
        
        logits = self.beta * ln + self.bias
        logits = torch.stack([-logits, logits], -1)
        
        self.logits.append(logits) 
    
    def sample_responses(self, b, t):

        cat = Categorical(logits=self.logits[-1])
       
        return cat.sample()

class SGFSocInf(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=120):
        
        na = 2  # number of choices
        ns = 2  # number of states
        no = 2  # number of outcomes
        
        super(SGFSocInf, self).__init__(runs, blocks, trials, na, ns, no)        

    def set_parameters(self, x=None):
        
        if x is not None:
            self.rho1 = x[..., 0].exp()
            self.h = (x[..., 1]-3).sigmoid()
            self.zeta = x[..., 2].sigmoid()
            self.beta = x[..., 3].exp()
            self.bias = x[..., 4]
        else:
            self.rho1 = .1
            self.h = .05
            self.zeta = .5
            self.beta = 10
        
        self.sig0 = ones(self.runs)
        self.mu0 = zeros(self.runs)
        self.theta0 = zeros(self.runs)
        self.rho2 = 10.
        
        self.npar = 5
        # set initial beliefs
        self.mu = [self.mu0]
        self.sig = [self.sig0]

        self.theta = [self.theta0]
        self.logits = []
        self.offers = []
        
    def update_beliefs(self, b, t, response_outcomes, mask=None):
        
        if mask is None:
            mask = ones(self.runs)
        
        o = (response_outcomes[-1][:, -2] + 1.)/2.

    	# 1st level
    	# Prediction
        muhat = self.mu[-1].sigmoid()

    	# Update the precision and the expectation conditioned on the absence of a change
        sig_pre = self.sig[-1]
        sig1 = (sig_pre + self.rho1) / (1 + (sig_pre + self.rho1) * mask * muhat * (1-muhat))
        mu1 = self.mu[-1] + sig1 * mask * (o - muhat)


    	# Update the precision and the expectation conditioned on the presence of a change
        sig2 =  self.rho2
        mu2 = mask * sig2 * (o - .5)

    	# marginal observation likelihood
        gamma = torch.sqrt(1 + sig_pre * 3.14159 / 8)
        muhat2 = (self.mu[-1] / gamma).sigmoid()
        l1 = muhat2**o * (1 - muhat2)**(1 - o)
        l2 = .5

    	# Update posterior jump probability
        theta_pre = self.h * (1-self.theta[-1])
        theta = l2 * theta_pre/( l1 * (1-theta_pre) +  l2 * theta_pre)

        sig = (sig1 * sig2)/((1  - theta) * sig2 + theta * sig1)
        self.mu.append(sig*(mu1*(1-theta)/sig1 + mu2*theta/sig2))
        self.theta.append(theta)
        self.sig.append(sig)

    def planning(self, b, t, offers):
        gamma = torch.sqrt(1 + (self.sig[-1] + self.rho1) * 3.14159 / 8)
        theta = self.theta[-1]
        
        b_soc = (self.mu[-1]/gamma).sigmoid() * (1 - theta) + theta/2
        
        b_vis = offers
        
        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        
        ln = b_int.log() - (1 - b_int).log()
        
        logits = self.beta * ln + self.bias
        logits = torch.stack([-logits, logits], -1)
        
        self.logits.append(logits)
    
    def sample_responses(self, b, t):

        cat = Categorical(logits=self.logits[-1])
       
        return cat.sample()
    
class BayesTempRevLearn(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=1000):
        
        na = 3  # number of choices
        ns = 2  # number of states
        no = 4  # number of outcomes
        
        super(BayesTempRevLearn, self).__init__(runs, blocks, trials, na, ns, no)
        
        self.nd = 100

    def set_parameters(self, x=None, set_variables=True, s0=0.):
        self.npar = 7
        if x is not None:
            self.mu = softplus(x[..., 0] + 20)
            self.tau = x[..., 1].exp()
            self.beta = x[..., 2].exp()
            self.lam = x[..., 3].sigmoid()
            self.s0 = x[..., 4].sigmoid()
            self.ph = (x[..., 5]+1.).sigmoid()*.5 + .5
            self.pl = (x[..., 6]-1.).sigmoid()*.5
        else:
            self.mu = 20*ones(self.runs)
            self.sigma = 25*ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.lam = ones(self.runs)
            self.ph = .8*ones(self.runs)
            self.pl = .2*ones(self.runs)
            self.s0 = .5*ones(self.runs)
               
        if set_variables:
            self.U = torch.stack([-ones(self.runs), ones(self.runs), 2 * self.lam - 1, 2 * self.lam - 1], dim=-1)
            self.set_prior_beliefs()
            self.set_state_transition_matrix()
            self.set_observation_likelihood()
            
            self.offers = []
            self.obs_entropy = []
            self.pred_entropy = []
            self.values = []
            self.logits = []
    
    def set_prior_beliefs(self):
        ps = torch.stack([self.s0, 1 - self.s0], -1)
        
        d = torch.arange(1., self.nd + 1.).reshape(1, -1)
        mu = self.mu.reshape(-1, 1)
        tau = self.tau.reshape(-1, 1)
        alpha = 1/tau
        theta = tau * mu
        
        lnd = d.log()
        
        self.pd = torch.softmax(- d / theta + lnd * (alpha - 1.), -1)
        
#        self.pd = torch.softmax(- .5 * tau * (lnd - mu)**2 - lnd, -1)
        
        P = self.pd.reshape(self.runs, 1, self.nd)*ps.reshape(self.runs, self.ns, 1)
        self.beliefs = [P]

    def set_state_transition_matrix(self):
        tm_dd = torch.diag(ones(self.nd-1), 1).repeat(self.runs, 1, 1)
        tm_dd[..., 0] = self.pd
        
        tm_ssd = zeros(self.ns, self.ns, self.nd)
        tm_ssd[..., 1:] = torch.eye(self.ns).reshape(self.ns, self.ns, 1)
        tm_ssd[..., 0] = (ones(self.ns, self.ns) - torch.eye(self.ns))/(self.ns - 1)
        
        self.tm_dd = tm_dd
        self.tm_ssd = tm_ssd
        
    def set_observation_likelihood(self):
        
        self.likelihood = zeros(self.runs, 2, self.na, self.ns, self.no)
        ph = self.ph
        pl = self.pl
        
        self.likelihood[:, :, -1, 0, 2] = 1.
        self.likelihood[:, :, -1, 1, 3] = 1.
        
        self.likelihood[:, 0, 0, 0, 0] = 1-ph
        self.likelihood[:, 0, 0, 0, 1] = ph
        
        self.likelihood[:, 0, 1, 1, 0] = 1-ph
        self.likelihood[:, 0, 1, 1, 1] = ph
        
        self.likelihood[:, 0, 1, 0, 0] = 1-pl
        self.likelihood[:, 0, 1, 0, 1] = pl
        
        self.likelihood[:, 0, 0, 1, 0] = 1-pl
        self.likelihood[:, 0, 0, 1, 1] = pl
        
        self.likelihood[:, 1, 0, 1, 0] = 1-ph
        self.likelihood[:, 1, 0, 1, 1] = ph
        
        self.likelihood[:, 1, 1, 0, 0] = 1-ph
        self.likelihood[:, 1, 1, 0, 1] = ph
        
        self.likelihood[:, 1, 0, 0, 0] = 1-pl
        self.likelihood[:, 1, 0, 0, 1] = pl
        
        self.likelihood[:, 1, 1, 1, 0] = 1-pl
        self.likelihood[:, 1, 1, 1, 1] = pl
        
        self.entropy = - torch.sum(self.likelihood*(self.likelihood+1e-16).log(), -1)
        
        
    def update_beliefs(self, b, t, response_outcomes, mask=None):
        
        if mask is None:
            mask = ones(self.runs)
        
        res = response_outcomes[0]
        out = response_outcomes[1]
        
        offers = self.offers[-1]
        
        prior = self.beliefs[-1]
        
        lik = 1. + mask.reshape(-1, 1) * (self.likelihood[range(self.runs), offers, res, :, out] - 1.)
        
        posterior = lik.reshape(self.runs, self.ns, 1) * prior
        norm = posterior.reshape(self.runs, -1).sum(-1).reshape(-1, 1, 1)
        
        prediction = torch.einsum('nij,klj,nlj->nki', self.tm_dd, self.tm_ssd, posterior/norm)

        self.beliefs.append(prediction)        


    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        self.offers.append(offers)
        subs = list(range(self.runs))
        
        lik = self.likelihood[subs, offers]
        entropy = self.entropy[subs, offers]

        psd = self.beliefs[-1]

        marginal = torch.einsum('naso,nsd->nao', lik, psd) 

        H = -torch.sum(marginal * (marginal + 1e-16).log(), -1)
        V = torch.einsum('nao,no->na', marginal, self.U)
        C = torch.einsum('nas,nsd->na', entropy, psd)
        
        self.values.append(V)
        self.obs_entropy.append(C)
        self.pred_entropy.append(H)
        
        self.logits.append(self.beta.reshape(-1, 1) * (V + H - C))   
    
    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()