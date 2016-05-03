# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:22:54 2016

Various experimental environments.

@author: Dimitrije Markovic
"""

import numpy as np

class Environment(object):
    """Base class for various environments."""
    def __init__(self, duration, dimension = 1, **kwargs):
        self.T = duration
        self.d = dimension
        self.observations = np.zeros((self.T, self.d))
        self.hidden_states = {}
        
    def generateObservations(self):
        for t in xrange(1, self.T):
            self.update_hidden_states(t)
            self.update_observations(t)
            
    def update_hidden_states(self, t, *args):
        """
        Updates hidden states given the corresponding transition distribution
        
        Parameters
        ----------
        t: int
        Current time step.
        """
        pass
    
    def update_observations(self, t, *args):
        """
        updates observations given the corresponding emission distribution.
        
        Parameters
        ----------
        t: int
        Current time step.
        """
        pass

class RewardEnv(Environment):
    """
    One dimensonal environment in which reward probability changes every time 
    step with probability p_r.
    """
    
class StateEnv(Environment):
    """
    One or two dimensional environment in which state of the environment
    changes with probability p_s. 
    """
    
class NoiseEnv(Environment):
    """
    One or two dimensional environment the emission variance 
    changes with probability p_n. 
    """
    
class StateNoiseEnv(StateEnv, NoiseEnv):
    """
    One or two dimensional environment in which state of the environment
    changes with probability p_s, and the emission variance changes with 
    probability p_n. 
    """
    
class AR2DEnv(Environment):
    """
    Two dimensional environment in which dynamics switches between two 
    AR(2) processes. First AR(2) process generates linear motion in a 
    2D plane, whereas second AR(2) process generates circular motion.
    Note: Do they both have to be AR(2)?
    """
    def __init__(self, duration, dimension = 2, emission_noise = 1., switch_probability = 0.05):
        self.sigma = emission_noise*np.eye(self.d)
        self.p = switch_probability
        self.As = [ 0.99*np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]
        #np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))
        
        self.hidden_states['active model'] = np.zeros(self.T).as_type(int)        
        
    def update_hidden_states(self, t):
        if(self.p >= np.random.rand()):
             self.hidden_states['active model'][t] = int(not  self.hidden_states['active model'][t-1])
        else:
             self.hidden_states['active model'][t] =  self.hidden_states['active model'][t-1]
             
    
    def update_observations(self, t):
        p = 2 #order of the AR process
        if( t < p):
            pass
        else:
            am = self.hidden_states['active model'][t]
            self.observations[t,:] = np.dot(self.As[am], self.observations[t-p:t, :].flatten())
                + np.dot(np.linalg.cholesky(self.sigma), np.random.randn(self.d))
             
                

