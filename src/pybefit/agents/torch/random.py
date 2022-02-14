#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:50:01 2018

@author: Dimitrije Markovic
"""

from torch import ones, zeros, arange
from torch.distributions import Categorical

from ..agent import Discrete

__all__ = [
        'Random'
]

class Random(Discrete):
    """Agent with discrete number of actions.
    """
    def __init__(self, pars, runs=1, blocks=1, trials=10):
        
        na = pars['na']  # number of choices
        
        super(Random, self).__init__(runs, blocks, trials, na, 1, 1)
                
        
    def set_parameters(self, *args, **kwargs):
        """Set free model parameters.
        """
        pass
        
    def update_beliefs(self, *args):
        """Update beliefs about hidden states given some sensory stimuli and action outcomes.
        """        
        pass

    def planning(self, *args, **kwargs):
        """Compute choice probabilities in current block and trial.
        """
        pass
        
    def sample_responses(self, *args):
        """Generate responses given response probability.
        """
        cat = Categorical(probs=ones(self.runs, self.na))
        
        return cat.sample()