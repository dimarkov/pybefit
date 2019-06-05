#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contain reinforcement learning agents for various experimental tasks
Created on Mon Jan 21 13:50:01 2019

@author: Dimitrije Markovic
"""

import torch
from torch import ones, zeros, arange
from torch.distributions import Bernoulli, Categorical

from .agent import Discrete

__all__ = [
        'RLSocInf',
        'RLTempRevLearn'
]

class RLSocInf(Discrete):
    
    def __init__(self, runs=1, blocks=1, trials=1):
        
        na = 2  #  number of actions
        ns = 2  #  number of states
        no = 2  #  number of outcomes
        super(RLSocInf, self).__init__(runs, blocks, trials, na, ns, no)
        
    def set_parameters(self, x=None):
        
        if x is not None:
            self.alpha = x[..., 0].sigmoid()
            self.zeta = x[..., 1].sigmoid()
            self.beta = x[..., 2].exp()
            self.bias = x[..., 3]
        else:
            self.alpha = .25*ones(self.runs)
            self.zeta = .95*ones(self.runs)
            self.beta = 10.*ones(self.runs)

        self.V0 = zeros(self.runs)
        self.npar = 4

        # set initial value vector
        self.values = [self.V0]
        self.offers = []
        self.logits = []
        
    def update_beliefs(self, b, t, response_outcomes, mask=None):
        
        if mask is None:
            mask = ones(self.runs)  
        
        V = self.values[-1]
        o = response_outcomes[-1][:, -2]
        
        # update choice values
        self.values.append(V + mask*self.alpha*(o - V))

    def planning(self, b, t, offers):
        """Compute response probability from values."""
        V = self.values[-1]
        b_soc = (1 + V)/2
        b_vis = offers
        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log()
        
        logits = self.beta * ln + self.bias
        logits = torch.stack([-logits, logits], -1)
        self.logits.append(logits)    

    def sample_responses(self, b, t):
        cat = Categorical(logits=self.logits[-1])
       
        return cat.sample()
    
class RLTempRevLearn(Discrete):
    """here we implement a reinforcement learning agent for the temporal 
    reversal learning task.    
    """
    
    def __init__(self, runs=1, blocks=1, trials=1):
        
        na = 3  #  number of actions
        ns = 2  #  number of states
        no = 4  #  number of outcomes
        super(RLTempRevLearn, self).__init__(runs, blocks, trials, na, ns, no)
        
    def set_parameters(self, x=None, set_variables=True):
        
        if x is not None:
            self.alpha = x[..., 0].sigmoid()
            self.kappa = x[..., 1].sigmoid()
            self.beta = x[..., 2].exp()
            self.bias = x[..., 3]
        else:
            self.alpha = .25*ones(self.runs)
            self.kappa = ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.bias = zeros(self.runs)
            
        if set_variables:
            self.V0 = zeros(self.runs, self.na)
            self.V0[:, -1] = self.bias
            self.npar = 4
        
            # set initial value vector
            self.values = [self.V0]
            self.offers = []
            self.outcomes = []
            self.logits = []
        
    def update_beliefs(self, b, t, response_outcome, mask=None):
        
        if mask is None:
            mask = ones(self.runs)
        
        V = self.values[-1]
        
        res = response_outcome[0]
        obs = response_outcome[1]
        
        hints = res == 2
        nothints = ~hints
        lista = arange(self.runs)
        
        # update choice values
        V_new = zeros(self.runs, self.na)
        V_new[:, -1] = self.bias
        
        if torch.get_default_dtype() == torch.float32:
            rew = 2.*obs[nothints].float() - 1.
        else:
            rew = 2.*obs[nothints].double() - 1.
        
        choices = res[nothints]
        l = lista[nothints]
        V1 = V[l, choices]
        V_new[l, choices] = V1 + self.alpha[nothints] * mask[nothints] * (rew - V1)
        
        V2 = V[l, 1 - choices]
        V_new[l, 1 - choices] = V2 - \
            self.alpha[nothints] * self.kappa[nothints] * mask[nothints] * (rew + V2)
        
        cue = obs[hints] - 2
        V_new[hints, cue] = 1.
        V_new[hints, 1 - cue] = - self.kappa[hints]
        self.values.append(V_new)

    def planning(self, b, t, offers):
        """Compute response probability from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        self.offers.append(offers)
        loc1 = offers == 0
        loc2 = ~loc1
        
        V = zeros(self.runs, self.na)
        V[loc1] = self.values[-1][loc1]
        if loc2.any():
            V[loc2, 0] = self.values[-1][loc2, 1]
            V[loc2, 1] = self.values[-1][loc2, 0]
            V[loc2, -1] = self.values[-1][loc2, -1]
        
        self.logits.append(self.beta.reshape(-1, 1) * V)    

    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)
       
        return cat.sample()