#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from torch.distributions import Categorical, Multinomial
from torch import ones, zeros

class SpaceAdventure(object):
    def __init__(self, conditions,
                 outcome_likelihoods=None,
                 init_states = None,
                 runs = 1,
                 mini_blocks = 1,
                 trials = 2):
        
        
        self.ns = 6 #number of states
        self.na = 2 #number of actions
        self.no = 5 #number of observations
        self.runs = runs # number of runs
        self.nmb = mini_blocks #number of mini-blocks for each run
        
        self.list = torch.arange(0, runs, 1, dtype=torch.long)
        
        if outcome_likelihoods is None:
            mnom = Multinomial(probs = ones(self.ns, self.no))
            self.ol = mnom.sample((runs, mini_blocks))
        else:
            self.ol = outcome_likelihoods
        
        self.conditions = conditions
        
        trans_prob = torch.tensor([.9, .5])
        self.tp = trans_prob[conditions[0]]
        
        self.rem_actions = conditions[1].clone()  # remaining number of actions
        
        self.make_transition_matrix()
        
        self.states = torch.empty(runs, mini_blocks, trials+1, dtype=torch.long)
        if init_states is not None:
            self.states[..., 0] = init_states
        else:
            cat = Categorical(probs = torch.ones(runs, mini_blocks, self.ns))
            self.states[..., 0] = cat.sample()
        
    def make_transition_matrix(self):
        p = self.tp  # transition probability
        na = self.na # number of actions
        ns = self.ns # number of states
        runs = self.runs # number of runs
        nmb = self.nmb # number of mini-blocks in each run
        
        self.tm = zeros(runs, nmb, na, ns, ns)
        
        # move left action - no tranistion uncertainty
        self.tm[..., 0, :-1, 1:] = torch.eye(ns - 1).repeat(runs, nmb, 1, 1)
        self.tm[..., 0, -1, 0] = 1
        
        # jump action - with varying levels of transition uncertainty
        self.tm[..., 1, -2:, 0:3] = (1 - p[..., None, None].repeat(1, 1, 2, 3))/2 
        self.tm[..., 1, -2:, 1] = p[..., None].repeat(1, 1, 2)
        self.tm[..., 1, 2, 3:6] = (1 - p[..., None].repeat(1, 1, 3))/2 
        self.tm[..., 1, 2, 4] = p
        self.tm[..., 1, 0, 3:6] = (1 - p[..., None].repeat(1, 1, 3))/2 
        self.tm[..., 1, 0, 4] = p
        self.tm[..., 1, 3, 0] = (1 - p)/2; self.tm[..., 1, 3, -2] = (1 - p)/2
        self.tm[..., 1, 3, -1] = p
        self.tm[..., 1, 1, 2:5] = (1 - p[..., None].repeat(1, 1, 3))/2 
        self.tm[..., 1, 1, 3] = p
    
    def update_environment(self, block, trial, responses):
        self.rem_actions[:, block] -= 1
        
        self.valid = self.rem_actions[:, block] > -1
        self.count = self.valid.sum()
        tm = self.tm[self.valid, block]
        res = responses[self.valid].long()
        states = self.states[self.valid, block, trial]
                
        state_prob = tm[range(self.count), res, states]
        assert state_prob.shape == (self.count, self.ns)
        cat = Categorical(probs=state_prob)
        
        self.states[self.valid, block, trial+1] = cat.sample()
        self.states[~self.valid, block, trial+1] = -1
        
    def sample_outcomes(self, block, trial):
        states = self.states[:, block, trial+1]
        
        outcome_prob = self.ol[:, block][self.list, states]
        cat = Categorical(probs = outcome_prob)
        
        sample = cat.sample()
        sample[~self.valid] = -1
        
        return sample
        