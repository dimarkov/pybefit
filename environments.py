# -*- coding: utf-8 -*-
"""Contains classes used to emulate various experimental environments.

Todo:
    * Write proper module level docstring.
"""
from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pd
import numpy as np

class Environment(object):
    """Abstract environment class. 
    
    All abstract methods in this class have to be implemented in any derived class.
    
    Attributes:
        T (int): Duration of the experimental block, i.e. number of observations.
        d_o (int): Dimension of the observation space.
        d_x (int): Dimension of the hidden state space.
        data (DataFrame): Data frame containg observation and hidden state values.
        
    Todo:
        * Check if abstract classes and their methods should have docstrings.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, T, **kwargs):
        """Basic constructor for the Environment class."""
        self.T = T
        if kwargs.has_key('d_o'):
            self.d_o = kwargs['d_o']
        else:
            self.d_o = 1
        
        if kwargs.has_key('d_x'):
            self.d_x = kwargs['d_x']
        else:
            self.d_x = 1
        
        self.data = pd.DataFrame()
        
    def generate_data(self):
        """Generates hidden states and observations."""
        for t in xrange(1, self.T+1):
            self.update_hidden_states(t)
            self.update_observations(t)
    
    @abstractmethod
    def expected_performance(self, *args, **kwargs):
        """Computes expected agents performance over the experimental block."""
        pass
            
    
    @abstractmethod        
    def update_hidden_states(self, t, *args):
        """Updates the hidden states.
        
        Args:
            t (int): Current time step.
        
        """
        pass
    
    @abstractmethod        
    def update_observations(self, t, *args):
        """Generates the stimuli (observations) in current hidden state.
        
        Args:
            t (int): Current time step.
        
        """
        pass


class BinomialEnvironment(Environment):
    """Environment with binary observations.
    
       Observation are binary values sampled from the binomial distribution with 
       dynamic expected value :math: 'p_t'. At each time step the expectations
       would either stay unchanged with probability :math: '1-\rho' or change 
       to a new value (drawn from an uniform distribution) with probability 
       :math: '\rho'.
       
       Attributes:
           rho (double): Probability of resampling the value of expectations
           p_t (array_like): Predifned time serries of expected values
       
       Todo: 
           * Imlement performance measure in this environment
       
    """
    
    def __init__(self, T, rho = 0.05, p_t = None):
        super(BinomialEnvironment, self).__init__(T)        
        self.rho = rho #switch_probability

        if p_t is not None:
            self.data[r'$p_t$'] = p_t #predefined time serries of expected values
        else:
            self.data[r'$o_t$'] = [np.nan]*(self.T + 1)
            self.data[r'$p_t$'] = [np.nan]*(self.T + 1)
            self.data[r'$p_t$'][0] = np.random.rand()

    
    def update_hidden_states(self, t):
        """Update the value of :math: 'p_t'.
        
        Args:
            t (int): Current time step.
            
        """
    
        if(np.random.binomial(1, self.rho)):
            self.data[r'$p_t$'][t] = np.random.rand()
        else:
            self.data[r'$p_t$'][t] = self.data[r'$p_t$'][t-1]
        
    def update_observations(self, t):
        """Sample an observation for the current time step. 
        
        Args:
            t (int): Current time step.
            
        """
        
        self.data[r'$o_t$'][t] = np.random.binomial(1, self.data[r'$p_t$'][t])
        
    def expected_performance(self, r_p):
        """ Returns the expected agents performance for the given probability 
        of selecting one of the two possible options.
        
        Args:
            r_p (array_like): Response probability
        
        """
        return np.sum( self.data[r'$p_t$']*r_p + (1- self.data[r'$p_t$'])*(1-r_p) )
    
class StateEnv(Environment):
    """
    One or two dimensional environment in which state of the environment
    changes with probability p_s. 
    """
    
    def __init__(self, duration, dimension = 1, switch_probability = 0.05,
                 observation_noise = 1., diffusion_rate = 0.01, 
                 perturbation_noise = 10.):
        self.p_s = switch_probability
        self.sigma = observation_noise
        self.w1 = diffusion_rate
        self.w2 = perturbation_noise
        self.hidden_states['state value'] = np.zeros(self.T, self.d).as_type(int)
        
    def update_hidden_states(self, t, *args):
        """
        The values of the hidden state follow a diffusion process with probability 
        1-p_s or switch to a new state value with probability p_s, where the new
        state is sampled from a zero mean normal distribution with variance w_2,. 
        
        Parameters
        ----------
        t: int
        Current time step.
        """
    
        if(self.p_s >= np.random.rand()):
            self.hidden_states['state value'][t, :] = \
                np.sqrt(self.w2)*np.random.randn(self.d)
        else:
            self.hidden_states['state value'][t] = \
                self.hidden_states['state value'][t-1] + \
                np.sqrt(self.w1)*np.random.randn(self.d)
            
    def update_observations(self, t, *args):
        """
        Generate an observation of the current state value corrapted by the 
        observation noise \sigma.
        
        Parameters
        ----------
        t: int
        Current time step.
        """
   
        state_val = self.hidden_states['state value'][t]
        self.observations[t] = state_val + np.sqrt(self.sigma)*np.random.randn(self.d)
        
class NoiseEnv(Environment):
    """
    One or two dimensional environment the emission variance 
    changes with probability p_n. 
    """
    
class StateNoiseEnv(Environment):
    """
    One or two dimensional environment in which state of the environment
    changes with probability p_s, and the emission variance changes with 
    probability p_n. 
    """
    
    pass
    
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
        ar_ord = 2 #order of the AR process
        if( t < ar_ord):
            pass
        else:
            am = self.hidden_states['active model'][t]
            self.observations[t,:] = np.dot(self.As[am], self.observations[t-ar_ord:t, :].flatten()) \
                + np.dot(np.linalg.cholesky(self.sigma), np.random.randn(self.d))
                
def main():
    import seaborn as sns
    sns.set(style = "white", palette="muted", color_codes=True)
    
    T = 100
    
    be = BinomialEnvironment(T)
    be.generate_data()
    ax = be.data.plot(y = r'$o_t$', style = 'go')
    ax = be.data.plot(y = r'$p_t$', style = 'k-', ax = ax)
    ax.legend(numpoints = 1)

if __name__ == "__main__":
    main()
             
                

