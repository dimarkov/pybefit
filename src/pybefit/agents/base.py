#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:50:01 2018

@author: Dimitrije Markovic
"""

__all__ = [
    'Discrete',
    'Continous'        
]

class Discrete(object):
    """Agent with discrete number of actions.
    """
    def __init__(self, runs, blocks, trials, na, ns, no):
        
        self.runs = runs
        self.nb = blocks
        self.nt = trials
        
        self.na = na  # number of actions
        self.ns = ns  # number of states
        self.no = no  # number of outcomes        
        
    def set_parameters(self, *args, **kwargs):
        """Set free model parameters.
        """
        raise NotImplementedError
        
    def update_beliefs(self, block, trial, **kwargs):
        """Update beliefs about hidden states given some sensory stimuli and action outcomes.
        """        
        raise NotImplementedError

    def planning(self, block, trial):
        """Compute choice probabilities in current block and trial.
        """
        raise NotImplementedError
        
    def sample_responses(self, block, trial):
        """Generate responses given response probability.
        """
        raise NotImplementedError



class Continous(Discrete):
    """Agent with continous number of actions.
    """
    def __init__(self, runs, blocks, trials, ns, no):
        super(Continous, self).__init__(runs, blocks, trials, -1, ns, no)