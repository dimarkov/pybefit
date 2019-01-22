#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  22 14:50:01 2018

@author: Dimitrije Markovic
"""

import torch
from torch.distributions import Categorical

ones = torch.ones
zeros = torch.zeros

"""This module contains various experimental environments used for testing 
human behavior."""

from torch import ones, zeros, empty, tensor
import torch

from numpy import load

__all__ = [
        'SocialInfluence']

class SocialInfluence(object):
    """Here we define the environment for the social learning task
    """
    
    def __init__(self, stimuli, nsub=1, blocks=1, trials=120):
        
        self.trials = trials
        self.nsub = nsub

        # set stimuli 
        self.stimuli = stimuli
        
    def update_environment(self, block, trial, responses):
        """Generate stimuli for the current block and trial
        """        
        outcomes = self.stimuli['reliability'][block, trial]
        offers = self.stimuli['offers'][block, trial]
        
        self.stimulus = {'outcomes': outcomes,
                      'offers': offers}
    
        
    def get_stimulus(self, *args):
        """Returns dictionary of all stimuli values relevant for update of agent's beliefs.
        """
        return self.stimulus
        