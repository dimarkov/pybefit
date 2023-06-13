#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains a base class for task environment

@author: Dimitrije Markovic
"""

class Task(object):
    
    def __init__(self, nsub, blocks, trials):
        
        self.blocks = blocks  # number of experimental blocks
        self.trials = trials  # number of trials
        self.nsub = nsub  # number of subjects

    def get_offer(self, block, trial, *args, **kwargs):
        """Define an offer for a current block, trial pair that defines a unique stimuli, that is choices for a given block trial pair"""
        
        return None
        
    def update_environment(self, block, trial, *args, **kwargs):
        """Generate stimuli for task's current block and trial
        """

        raise NotImplementedError    
        