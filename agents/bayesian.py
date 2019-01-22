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

__all__ = [
        'HGFSocInf',
        'SGFSocInf'
]

class HGFSocInf(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=120):
        
        na = 2  # number of choices
        ns = 2  # number of states
        no = 2  # number of outcomes
        
        super(HGFSocInf, self).__init__(runs, blocks, trials, na, ns, no)
        
    def set_parameters(self, x=None):
        
        
        if x is not None:
            self.mu0 = zeros(self.runs, 2)
            self.mu0[:, 0] = x[..., 0]
            self.mu0[:, 1] = - x[..., 1].relu() - 2.
            self.pi0 = 1. + x[..., 2:4].exp()
            self.kappa = x[..., 4].sigmoid()
            self.zeta = x[..., 5].sigmoid()
            self.beta = x[..., 6].exp()
        else:
            self.mu0 = zeros(self.runs, 2)
            self.pi0 = ones(self.runs, 2) 
            self.kappa = .1*ones(self.runs)
            self.zeta = .95*ones(self.runs)
            self.beta = 10.*ones(self.runs)
        
        self.eta = .1
                
        #set initial beliefs
        self.mu = [self.mu0]
        self.pi = [self.pi0]
        
        self.npars = 7
        self.offers = []
        self.logprobs = []

    def update_beliefs(self, b, t, outcomes=None, offers=None, masks=1):
        
        self.offers.append(offers)

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

        pi1 = pihat1 + masks * muhat * (1-muhat)
        wda = masks * ( (outcomes + 1)/2 - muhat) / pi1
        mu.append(mu_pre[:, 0] + wda)
        pi.append(pi1)


        # Volatility prediction error
        da1 = masks * ((1 / pi1 + (wda)**2) * pihat1 - 1)

        # 3rd level
        # Precision of prediction
        pihat2 = pi_pre[:, 1].clone() / (1. + pi_pre[:, 1] * self.eta)

        # Weighting factor
        w2 = w1 * pihat1 * masks

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

    def planning(self, b, t):
        b_soc = self.mu[-2][:, 0].sigmoid()
        b_vis = self.offers[-1]
        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log()
        self.logprobs.append(self.beta * ln) 
    
    def sample_responses(self, b, t):
        logits = self.logprobs[-1]
        bern = Bernoulli(logits=logits)
       
        return bern.sample()

class SGFSocInf(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=120):
        
        na = 2  # number of choices
        ns = 2  # number of states
        no = 2  # number of outcomes
        
        super(SGFSocInf, self).__init__(runs, blocks, trials, na, ns, no)        

    def set_parameters(self, x=None):
        
        if x is not None:
            self.mu0 = x[..., 0]
            self.sig0 = x[..., 1].exp()
            self.rho1 = x[..., 2].exp()
            self.h = (x[..., 3]-3).sigmoid()
            self.zeta = x[..., 4].sigmoid()
            self.beta = x[..., 5].sigmoid()
        else:
            self.mu0 = zeros(self.runs)
            self.sig0 = ones(self.runs)
            self.rho2 = 10.
            self.h = .05
            self.zeta = .5
            self.beta = 10
        
        self.theta0 = zeros(self.runs)
        self.rho2 = 10.
        
        self.npars = 6
        # set initial beliefs
        self.mu = [self.mu0]
        self.sig = [self.sig0]

        self.theta = [self.theta0]
        self.logprobs = []
        self.offers = []
        
    def update_beliefs(self, b, t, outcomes=None, offers=None, masks=1):
        
        self.offers.append(offers)
        o = (outcomes + 1.)/2.

    	# 1st level
    	# Prediction
        muhat = self.mu[-1].sigmoid()

    	# Update the precision and the expectation conditioned on the absence of a change
        sig_pre = self.sig[-1]
        sig1 = (sig_pre + self.rho1) / (1 + (sig_pre + self.rho1) * masks * muhat * (1-muhat))
        mu1 = self.mu[-1] + sig1 * masks * (o - muhat)


    	# Update the precision and the expectation conditioned on the presence of a change
        sig2 =  self.rho2
        mu2 = masks * sig2 * (o - .5)

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

    def planning(self, b, t):
        gamma = torch.sqrt(1 + self.sig[-2]*3.14159/8)
        b_soc = (self.mu[-2]/gamma).sigmoid()
        b_vis = self.offers[-1]
        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log()
        self.logprobs.append(self.beta * ln)
    
    def sample_responses(self, b, t):
        logits = self.logprobs[-1]
        bern = Bernoulli(logits=logits)
       
        return bern.sample()