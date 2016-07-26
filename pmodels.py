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
        stimuli (DataFrame): Time series of sensory stimuli
        T (int): Number of observations
        
    """
    
    __metaclas__ = ABCMeta
    
    def __init__(self, stimuli):
        self.stimuli = stimuli #set the time series of the sensory stimuli
        self.T = len(stimuli) - 1
    
    def posterior_beliefs(self, *args):
        """Estimate the posterior beliefs over an experimental block """
        
        for t in range(1,self.T+1):
            self.update_beliefs(t, *args)
            
    def get_free_energy(self, *args):
        """Estimate the total variational free-energy of the perceptual model."""
        
        fe = 0
        for t in range(1, self.T+1):
            self.update_beliefs(t, *args)
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
       
    def __init__(self, stimuli, mu0 = .5):
        super(RescorlaWagner, self).__init__(stimuli)
    
        self.posterior = pd.DataFrame()
        self.posterior[r'$\mu_t$'] = [np.nan]*(self.T + 1)
        self.posterior[r'$\mu_t$'][0] = mu0
        self.posterior[r'$v_t$'] = [np.nan]*(self.T + 1)
        
    def update_beliefs(self, ind, alpha, *args):
        """Estimate the updated beliefs after observing a stimulus at trial ind."""

        self.posterior[r'$v_t$'][ind] = 1/alpha
        self.posterior[r'$\mu_t$'][ind] = self.posterior[r'$\mu_t$'][ind-1] + \
                                        alpha*( self.stimuli[r'$o_t$'][ind] - \
                                        self.posterior[r'$\mu_t$'][ind-1])
                                        
    def free_energy(self, ind, *args):
        """Compute the free energy of the perceptual model at trial ind.
        
        Args:
            ind (int): Trial index.
        
        Returns:
            fe (double): Free energy of the perceptual model at given trial.
        
        """

        p = self.posterior[r'$\mu_t$'][ind-1]
        o = self.stimuli[r'$o_t$'][ind]
        
        if p == 0:
            if o:
                return -100000
            else:
                return 0
        elif p == 1:
            if o:
                return 0
            else:
                return -100000
        else:
            return o*np.log(p) + (1-o)*np.log(1-p)  

def main():
    
    from environments import BinomialEnvironment
    import seaborn as sns
    sns.set(style = "white", palette="muted", color_codes=True)
    
    T = 100
    be = BinomialEnvironment(T)
    be.generate_data()
    
    pm = RescorlaWagner(be.data)
    
    #use isres for optimisation of one dimensional functions
    from optmethods import isres
    bounds = {'ub': np.array([.999]), 'lb': np.array([.001])}
    f_opt, x_opt, res = isres( pm.get_free_energy, 1, 1e-6, 1e-8, bounds, np.array([0.5]) )
    print(f_opt, x_opt, res)
    
    pm.posterior_beliefs(x_opt)
    
    ax = be.data.plot(y = r'$o_t$', style = 'go')
    ax = be.data.plot(y = r'$p_t$', style = 'k--', ax = ax)
    
    ax = pm.posterior.plot(y = r'$\mu_t$', style = 'r-', ax = ax)
    ax.legend(numpoints = 1)
    
    #optimize preceptual surprise over multiple experimental blocks
    
    n = 100
    exp_blocks = np.empty_like(np.ones(n), dtype = object)
    for i in range(n):
        exp_blocks[i] = BinomialEnvironment(T)
        exp_blocks[i].generate_data()
        
    def total_fe(x, n_pars, blocks):
        fe = 0
        for b in blocks:
            pm = RescorlaWagner(b.data)
            fe += pm.get_free_energy(x)
            
        return fe
        
    fe = lambda x,p: total_fe(x, p, exp_blocks)
    f_opt, x_opt, res = isres( fe, 1, 1e-6, 1e-8, bounds, np.array([0.5]) )
    print(f_opt/n, x_opt, res)
    
    pm.posterior_beliefs(x_opt)
    
    ax = be.data.plot(y = r'$o_t$', style = 'go')
    ax = be.data.plot(y = r'$p_t$', style = 'k--', ax = ax)
    
    ax = pm.posterior.plot(y = r'$\mu_t$', style = 'r-', ax = ax)
    ax.legend(numpoints = 1)
    
    
    
if __name__ == '__main__':
    main()