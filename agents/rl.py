#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contain reinforcement learning agents for various experimental tasks
Created on Mon Jan 21 13:50:01 2019

@author: Dimitrije Markovic
"""

from torch import ones, zeros
from torch.distributions import Bernoulli

from .agent import Discrete

__all__ = [
        'RLSocInf'
]

class RLSocInf(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=1):
        
        na = 2  #  number of actions
        ns = 2  #  number of states
        no = 2  #  number of outcomes
        super(RLSocInf, self).__init__(runs, blocks, trials, na, ns, no)
        
    def set_parameters(self, x=None):
        
        if x is not None:
            self.V0 = 2*x[..., 0].sigmoid() - 1
            self.alpha = x[..., 1].sigmoid()
            self.zeta = x[..., 2].sigmoid()
            self.beta = x[..., 3].exp()
            self.bias = x[..., 4]
        else:
            self.V0 = zeros(self.runs)
            self.alpha = .25*ones(self.runs)
            self.zeta = .95*ones(self.runs)
            self.beta = 10.*ones(self.runs)
        
        self.npars = 5

        # set initial value vector
        self.values = [self.V0]
        self.offers = []
        self.logprobs = []
        
    def update_beliefs(self, b, t, outcomes=None, offers=None, masks=1):
        
        self.offers.append(offers)
        V = self.values[-1]
        # update choice values
        self.values.append(V + masks*self.alpha*(outcomes - V))

    def planning(self, b, t):
        """Compute response probability from values."""
        
        V = self.values[-2]
        b_soc = (1 + V)/2
        b_vis = self.offers[-1]
        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log() 
        self.logprobs.append(self.beta * ln + self.bias)    

    def sample_responses(self, b, t):
        logits = self.logprobs[-1]
        bern = Bernoulli(logits=logits)
       
        return bern.sample()