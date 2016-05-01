# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:22:54 2016

Various experimental environments.

@author: Dimitrije Markovic
"""

class Environment(object):
    """Base class for various environmens."""
    def __init__(duration, dimension, **kwargs):
        self.T = duration
        self.d = dimension
        self.observations = np.zeros((self.T, self.d))
        self.hidden_states = {}
        
    def generateObservations():
        for t in xrange(self.T):
            self.update_hidden_states(t)
            self.update_observations(t)
            
    def update_hidden_states(t):
        """Define transitions between hidden states"""
        pass
    
    def update_observations(t):
        pass
            
    
    
        

