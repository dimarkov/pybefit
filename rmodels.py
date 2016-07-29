# -*- coding: utf-8 -*-
"""This module contains various response models.

"""
from __future__ import division, print_function
from builtins import range
from abc import ABCMeta, abstractmethod, abstractproperty
import pandas as pd
import numpy as np


class ResponseModel(object):
    """Base class for response models.
    
    Attributes:
        prior (array_like): Prior expectation and uncertainty over model paramers.
        pm (PerceptualModel): Perceptual model.
        rp_t (array_like): Response probability.
        responses (array_like): Series of behavioral responses
        
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, prior, perceptual_model, d_r, responses):
        """
        Args:
            prior (array_like): Prior expectation and uncertanty over model parameters.
            perceptual_model (PerceptualModel): Perceptual model.
            d_r (int): dimensionality of responses
            responses (array_like): Series of behavioral responses
        """
        
        self.prior = prior
        self.pm = perceptual_model
        self.T = self.pm.T
        self.rp_t = np.zeros( (self.T + 1, d_r) )
        self.responses = responses
	
    @abstractmethod
    def get_responses(self, *args):
        """Generates behavioral responses."""
        pass
    
    @abstractmethod
    def estimate_response_probability(self, ind, **kwargs):
        """Estimates response probability at trial ind."""
        pass
    
    @abstractmethod 
    def sample_from_prior(self, *args, **kwargs):
        """Sample parameter values from the prior distribution."""
        pass
    
    @abstractmethod
    def get_total_log_likelihood(self, x, **kwargs):
        """Returns the total log-likelihoods over all trials.
        
        Args:
            x (array_like): Parameter values
        """
        pass

    @abstractmethod
    def get_response_probability(self, ind):
        """Returns the response probability at trial ind."""
        pass
    
    @abstractmethod
    def get_response(self, ind, **kwargs):
        """Returns the response at trial ind."""
        pass
    
    @abstractmethod
    def get_log_likelihood(response_probability, response):
        """Return the marginal log-likelihood at trial ind"""
        pass

class SoftMaxResponses(ResponseModel):
    """Responses are generated from the softmax over posterior beliefs.
    
    
    """
    
    def __init__(self, prior, perceptual_model, d_r, responses = None):
        super(SoftMaxResponses, self).__init__(prior, perceptual_model, d_r, responses)

    def get_responses(self, pm_kwargs, rm_kwargs):
        """Generate responses from the response probability.
        Args:
            pm_kwargs (dict): Parameters of the perceptual model.
            rm_kwargs (dict): Parameters of the response model.
        """
        if self.responses is None:
            self.responses = np.zeros(self.T + 1, dtype = np.int)
            self.responses[0] = -1
            for t in range(1, self.T+1):
                if self.rp_t[t-1].sum() == 0:
                    self.estimate_response_probability(t-1, pm_kwargs = pm_kwargs,
                                                       rm_kwargs = rm_kwargs)
                
                #sample new response
                self.responses[t] = self.get_response(t-1, sample = True)
                
                #generate new observation from that response
                self.pm.env.generate_observations(t, self.responses[t] )
            
        return pd.DataFrame({r'$r_t$': self.responses})
    
    def get_response(self, ind, sample = False):
        if sample:
            #generate response form response probability
            return np.random.multinomial(1, self.rp_t[ind-1]).argmax()
        else:
            #return the recorded response
            return int(self.pm.env.o_t[ind, 1])
            
        
        
    def estimate_response_probability(self, ind, pm_kwargs = {}, rm_kwargs = {}):
        """
        Args:
            ind (int): Trial index
            pm_kwargs (dict): Dictionary containing parameters of the perceptual model.
            rm_kwargs (dict): Dictionary containing parameters of the response model.
            stimuli (array_like): Stimuli presented to the agent at trial ind
        """
        
        if not 'alpha' in pm_kwargs.keys():
            #check if posterior beliefs are estimated
            if self.pm.posterior[ind].sum() == 0:
                #TODO: implement sample from prior
                print("Parameters of the perceptual model are sampled from the prior distribution.")
                pm_pars = self.sample_from_prior()
                self.pm.update_beliefs(ind, **pm_kwargs)
        else:
            #if one provides pm_kwargs the beliefs will be estimated again
            if ind > 0:
                self.pm.update_beliefs(ind, **pm_kwargs)
            else:
                if 'prior' in pm_kwargs.keys():
                    self.pm.posterior[ind] = pm_pars['prior']
        

        if 'theta' in rm_kwargs.keys():
            self.rp_t[ind] = rm_kwargs['theta']*np.log( self.pm.posterior[ind] )
            self.rp_t[ind] = np.exp( self.rp_t[ind] - np.max(self.rp_t[ind]) )
        else:
            self.rp_t[ind] = self.pm.posterior[ind]
            
        self.rp_t[ind] /= self.rp_t[ind].sum()
        
        return self.rp_t[ind]
        
    def get_response_probability(self, ind):
        """Returns the response probability at trial ind."""
        return self.rp_t[ind]
            
    def sample_from_prior(self):
        """Sample parameter values from the prior distribution."""
        raise NotImplementedError
    
    def get_total_log_likelihood(self, x, transform = True):
        
        if transform:
            pm_kwargs = {'alpha': 1/(1+np.exp(-x[0])), 'beta': 1/(1+np.exp(-x[1]))}
            rm_kwargs = {'theta': np.exp(x[2])}
        else:
            pm_kwargs = {'alpha': x[0], 'beta': x[1]}
            rm_kwargs = {'theta': x[2]}
        
        tll = 0
        for t in range(0, self.T):
            self.estimate_response_probability(t, pm_kwargs = pm_kwargs, 
                                               rm_kwargs = rm_kwargs)
            
            response_probability = self.get_response_probability(t)
            response = self.get_response(t+1)
            tll += self.get_log_likelihood(response_probability, response )
        
        return tll    
    
    @staticmethod
    def get_log_likelihood(response_probability, observed_response):
        """Estimate the log-likelihood of the observed response at trial ind."""
        
        return np.log(response_probability[observed_response])
        

def main():
    import time
    from environments import MultiArmedBandit
    from pmodels import RescorlaWagner
    from inference import MLEInference
    
    def expected_performance(x):
        """Example of how the expected performance can be computed."""
        pm_pars = {'alpha': x[0], 'beta': x[1]}
        rm_pars = {'theta': x[2]}
        
        T = 100 #number of trials
        n_b = 2 #number of bandits
        rho = 0.01 #switch probability of the arm-reward contingencies
        
        n_env = 100 #number of environments
        n_blocks = 1 #number of experimental blocks
        ep = 0 # expected performance
        
        for n in range(n_env):
            #generate n_env mutli-armed bandit environmets
            env = MultiArmedBandit(T, rho = rho, n_b = n_b)
            pm = RescorlaWagner(env, n_b)
            for m in range(n_blocks):
                #in each environment repeat the experiment n_blocks times
                rm = SoftMaxResponses([], pm, d_r)
                rm.get_responses(pm_pars, rm_pars)
                #For each block compute the expected performance.
                ep += env.expected_performance()
        
        return ep/(n_env*n_blocks)

    
    T = 100 #number of trials
    n_b = 2 #number of bandits
    rho = 0.01 #switch probability of the arm-reward contingencies
    d_r = n_b
    
    ###########################################################################
    #Lets try to find the set of parameters that lead to the highest performance.
    #This takes lots of time, as it converges very slowly because of noisy estimates.
    #Just comment it out and use the x_opt values provided bellow.
    
#    from optmethods import cmaes
#    n_p = 3
#    bounds = bounds = {'ub': np.array([1., 1., 100.]), 'lb': np.zeros(3)}
#    f_opt, x_opt, res_msg = cmaes( expected_performance, n_p, 1e-2, 1e-4, 
#                                  bounds, np.zeros(n_p), verb_disp = 10 )
#    print(f_opt, x_opt, res_msg)
    
    #the following values give resonably high expected performance
    x_opt = np.array([0.125, 0.1, 10])
    ###########################################################################    
    
    t_start = time.time()
    print( expected_performance(x_opt), time.time() - t_start )
    
    ####mle estimate of the parameter values#############################
    
    #we first simulate the behavior
    env = MultiArmedBandit(T, rho = rho, n_b = n_b)
    d_b = n_b
    pm = RescorlaWagner(env, d_b)
    d_r = n_b
    rm = SoftMaxResponses( [], pm, d_r)
    
    pm_pars = {'alpha': x_opt[0], 'beta': x_opt[1]} #parameters of the perceptual model
    rm_pars = {'theta': x_opt[2]} # parameters of the response model
    rm.get_responses(pm_pars, rm_pars)
    
    pm_inference = RescorlaWagner(env, d_b)
    rm_inference = SoftMaxResponses([], pm_inference, d_r)
    
    opts = {'np': 3, 'verb_disp':100}
    
    mle = MLEInference(opts = opts)
    m_mle, s_mle, f_mle = mle.infer_posterior(rm_inference)    
    
    p_mle = [1/(1+np.exp(-m_mle[:-1])), np.exp(m_mle[-1])]
    print(f_mle, p_mle, np.diag(s_mle))
    ###########################################################################

if __name__ == '__main__':
    main()

    
    
