#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains various experimental environments used for testing 
human behavior.

Created on Thu Feb  22 14:50:01 2018

@author: Dimitrije Markovic
"""
import torch
from torch import zeros, ones
from torch.distributions import Categorical, Multinomial, Dirichlet

__all__ = [
        'MultiArmedBandit'
]

class MultiArmedBandit(object):
    """Implementation of a multi-armed bandit task. Bandit has K arms, each associated 
       with specific probability of delivering different outcomes.
    """
    
    def __init__(self, priors, transitions, context, offers, arm_types, states=None, nsub=1, blocks=1, trials=100):
        
        self.priors = priors  # prior probabilities
        self.tms = transitions  # dictionary containing tranistion matrices
        
        self.blocks = blocks
        self.trials = trials
        self.nsub = nsub
        
        if states is not None:
            self.states = states
            self.fixed_states = 1
        else:
            self.fixed_states = 0
        
        self.arm_types = arm_types
    
        self.context = context        
        
        self.offers = offers
        
        self.initialise()
    
    def initialise(self):
        if not self.fixed_states:
            blocks = self.blocks
            trials = self.trials
            nsub = self.nsub
            
            ns, nf = self.priors['probs'].shape
            self.states = {'points': zeros(blocks, trials+1, nsub, nf, dtype=torch.long),
                           'probs': zeros(blocks, trials+1, nsub, ns, nf),
                           'locations': zeros(blocks, trials+1, nsub, dtype=torch.long) }
        
        return self
    
    def get_offers(self, block, trial):
        
        if trial == 0:
            self.update_states(block, trial)
            return {'locations': self.states['locations'][block, trial], 
                'points': self.states['points'][block, trial]}
            
        else:
            return {'locations': self.states['locations'][block, trial], 
                'points': self.states['points'][block, trial]}
    
    def update_states(self, block, trial, responses=None, outcomes=None):
        
        if trial == 0:
            self.states['points'][block, trial] = 0
            
            if block == 0:
                probs =  self.priors['probs']
                self.states['probs'][block, trial] = probs
            else:
                self.states['probs'][block, trial] = self.states['probs'][block-1, -1]
                
            self.states['locations'][block, trial] = Categorical(probs=self.priors['locations']).sample((self.nsub,))
        else:
            self.states['points'][block, trial] = self.states['points'][block, trial - 1] + outcomes.long()
            self.states['probs'][block, trial] = self.states['probs'][block, trial - 1]
            loc = self.states['locations'][block, trial - 1]
            self.states['locations'][block, trial] = \
                Categorical(probs=self.tms['locations'][responses, loc]).sample()
        
        if trial < self.trials:
            return zeros(self.nsub)
        else:
            success = torch.any(self.states['points'][block, trial, :, 1:] > 2*self.trials//3, -1)
            return success.long()
        
    def update_environment(self, block, trial, responses):
        """Generate stimuli for the current block and trial and update the state
        """

        # offers in the current trial
        offers = self.offers[block][trial]
        
        # selected arm types
        arm_types = self.arm_types[offers, responses]

        # each selected arm is associated with specific set of reward probabilities
        probs = self.states['probs'][block, trial, range(self.nsub), arm_types]
        out1 = Multinomial(probs=probs).sample()

        out = {'locations': responses,
               'features': out1.argmax(-1)}        
        
        out2 = self.update_states(block, trial+1, responses=responses, outcomes=out1)
            
        return [responses, (out, out2)]
    
        