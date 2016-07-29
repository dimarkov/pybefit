# -*- coding: utf-8 -*-
"""These module contains various perceptual models.

"""
from __future__ import division, print_function
from builtins import range
from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pd
import numpy as np

class PerceptualModel(object):
    """Base class for perceptual models.
    
    Attributes:
        T (int): Number of trials
        d (int): Dimensionality of the posterior distribution
        env (Environment): Experimental environment of the perceptual model.
        
    """
    
    __metaclas__ = ABCMeta
    
    def __init__(self, env, d):
        self.env = env #set the time series of the sensory stimuli
        self.T = self.env.T
        self.d = d
        
    def posterior_beliefs(self, *args):
        """Estimate the posterior beliefs over an experimental block."""
        
        for t in range(1,self.T+1):
            self.update_beliefs(t, *args)
            
    def get_free_energy(self, *args, **kwargs):
        """Estimate the total variational free-energy of the perceptual model."""
        
        fe = 0
        for t in range(1, self.T+1):
            self.update_beliefs(t, *args, **kwargs)
            fe += self.free_energy(t, *args)
        
        return fe
    
    @abstractmethod    
    def free_energy(self, ind, *args):
        """Compute the free energy of the perceptual model at trial ind."""
        pass
    
    @abstractmethod
    def update_beliefs(self, ind, *args):
        """Estimate the updated beliefs after observing a stimulus at trial ind"""
        pass


class RescorlaWagner(PerceptualModel):
    """Perceptual model based on the Rescorla-Wagner delta rule.
    
       The update equation of this perceptual model follow the Rescorla-Wagner
       learning rule defined as :math: '\mu_t = \mu_{t-1} + \alpha (o_t - \mu_{t-1})'. 
       Learning rate :marh: '\alpha' is the only free parameter of the model.
       
       Attributes:
           prior (array_like): Prior expectation and uncertainty.
           posterior (DataFrame): Time series of the suficient statistics of the posterior distribution.
       
    """
       
    def __init__(self, env, d, mu0 = .5):
        """
        Args:
            env (Environment): Experimental environment of the perceptual model.
            d (int): Dimensionality of the posterior distribution
            mu0 (double): Prior expectation over hidden states
        """
        super(RescorlaWagner, self).__init__(env, d)
    
        self.posterior = np.zeros( (self.T+1, self.d) )
        self.posterior[0, :] = mu0
        
    def get_beliefs(self, alpha = None):
        
        update = self.posterior[1:, :].sum() == 0
        
        if alpha is None:
            if update:
               alpha = .5
               for t in range(1, self.T + 1):
                   self.update_beliefs(t, alpha)
        else:
            for t in range(1, self.T + 1):
                self.update_beliefs(t, alpha)

        if self.d == 1:
            return pd.DataFrame({r'$\mu_t$': self.posterior[:, 0]})
        else:
            return pd.DataFrame(self.posterior, columns = [r'$\mu_{t,%d}$' % i 
                            for i in range(self.d)])
        
        
    def update_beliefs(self, ind, alpha, *args, **kwargs):
        """Estimate the updated beliefs after observing a stimulus at trial ind."""
        
        self.posterior[ind, :] = self.posterior[ind-1, :]        
        
        #forgeting term
        if 'beta' in kwargs.keys():
            beta = kwargs['beta']
        else:
            beta = 0
        
        self.posterior[ind, :] += beta*( .5 - self.posterior[ind-1, :] )
        
        if self.env.o_t[ind, 1] == -1:
            obs, choice = self.env.generate_observations(ind)
        else:
            obs, choice = self.env.o_t[ind, :].astype(int)
        
        self.posterior[ind, choice] += alpha*( obs - self.posterior[ind-1, choice] )
                                        
    def free_energy(self, ind, *args):
        """Compute the free energy of the perceptual model at trial ind.
        
        Args:
            ind (int): Trial index.
        
        Returns:
            fe (double): Free energy of the perceptual model at given trial.
        
        """

        mu = self.posterior[ind-1, :]
        obs, choice = self.env.o_t[ind, :].astype(int)
        
        if mu[choice] == 0:
            if obs:
                return -100000
            else:
                return 0
                
        elif mu[choice] == 1:
            if obs:
                return 0
            else:
                return -100000
        else:
            return obs*np.log(mu[choice]) + (1-obs)*np.log(1-mu[choice])  

def main():
    
    from environments import MultiArmedBandit
    import seaborn as sns
    sns.set(style = "white", palette="muted", color_codes=True)
    
    T = 100
    env = MultiArmedBandit(T)
    pm = RescorlaWagner(env, env.d_x)
    
    obs = env.get_observations()
    hst = env.get_hidden_states()
    
    #use isres for optimisation of one dimensional functions
    from optmethods import isres
    bounds = {'ub': np.array([1.]), 'lb': np.array([0.])}
    f_opt, x_opt, res = isres( pm.get_free_energy, 1, 1e-6, 1e-8, bounds, np.array([0.5]) )
    print(f_opt, x_opt, res)
    
    post = pm.get_beliefs(alpha = x_opt)

    
    ax = obs.plot(y = r'$o_t$', style = 'go')
    ax = hst.plot(y = r'$p_t$', style = 'k--', ax = ax)
    
    ax = post.plot(y = r'$\mu_t$', style = 'r-', ax = ax)
    ax.legend(numpoints = 1)
    
    #optimize preceptual surprise over multiple experimental blocks
    
    def total_fe(x, n_pars, blocks):
        fe = 0
        for b in blocks:
            pm = RescorlaWagner(b, b.d_x)
            fe += pm.get_free_energy(x)
            
        return fe
    
    n = 100
    T = 100
    exp_blocks = [MultiArmedBandit(T)]*100 
        
    fe = lambda x,p: total_fe(x, p, exp_blocks)
    f_opt, x_opt, res = isres( fe, 1, 1e-6, 1e-8, bounds, np.array([0.5]) )
    print(f_opt/n, x_opt, res)
    
    post = pm.get_beliefs(alpha = x_opt)
    
    ax = obs.plot(y = r'$o_t$', style = 'go')
    ax = hst.plot(y = r'$p_t$', style = 'k--', ax = ax)
    
    ax = post.plot(y = r'$\mu_t$', style = 'r-', ax = ax)
    ax.legend(numpoints = 1)
    
    
    
if __name__ == '__main__':
    main()