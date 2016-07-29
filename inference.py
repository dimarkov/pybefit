# -*- coding: utf-8 -*-
"""This module implements the class that performs meta-Bayesian inference
"""
from __future__ import division, print_function
from builtins import range
import sys
from abc import ABCMeta, abstractmethod
import optmethods
import numpy as np
from numdifftools import Hessian

class Inference(object):
    """Base class for various inference methods"""
    __metaclass__ = ABCMeta
    
    def __init__(self, optimization_method = 'cmaes', 
                 ftol = 1e-8, xtol = 1e-12, 
                 bound = 10, **kwargs):
        
        #set optimization method; possible choices are 'cmaes' and 'isres'
        self.optimization_method = optimization_method
        
        #set search bounds
        if 'bounds' in kwargs.keys():
            self.bounds = {'lb': kwargs['bounds'][0,:],
                           'ub': kwargs['bounds'][1,:]}
        else:
            self.bounds = {'lb': -bound, 'ub': bound}
                
        #set function tolerance
        self.ftol = ftol
        
        #set parameter value tolerance
        self.xtol = xtol
        
        if 'opts' in kwargs.keys():
            self.opts = kwargs['opts']
    
    @abstractmethod
    def infer_posterior(self, response_model):
        pass


class LAinference(Inference):
    """
    Laplace approximation based estimate of the posterior 
    parameter distribution of various behavioral models.
    """
    
    def __init__(self, **kwargs):
        super(LAinference, self).__init__(**kwargs)
        
    def infer_posterior(self, rmodel, options):
        pass


class MLEInference(Inference):
    """Maximum likelihod estiamte of the free model parameter"""
    
    def __init__(self, **kwargs):
        super(MLEInference, self).__init__(**kwargs)
            
    def infer_posterior(self, rm):
        """Finds the maximum likelihood estimate of the free model parameters.
        Args:
            rm (ResponseModel): Response model used to infer the parameter values.
        """
        
        f_opt = -np.inf #model log-evidence
        nump = self.opts.pop('np') #number of parameters
        m_opt = np.zeros(nump) #posterior expectation
        s_opt = np.inf*np.eye(nump) #posterior covariance matrix
        
        #set the optimization method
        opt_method = getattr(optmethods, self.optimization_method)
        
        count_errors = 0
        count_modes = 0
        count_runs = 0
        while(True):
            x_init = self.bounds['lb'] + np.random.rand(nump)* \
                (self.bounds['ub'] - self.bounds['lb'])
            
            try:
                f_opt_tmp, m_opt_tmp, stopflag = opt_method(rm.get_total_log_likelihood, 
                                                            nump, self.ftol, self.xtol, 
                                                            self.bounds, x_init,
                                                            **self.opts)
            except:
                print( "Error in optimization procedure \n" )
                print( "Unexpected error:", sys.exc_info()[0], '\n' )
                print( "Value: ", sys.exc_info()[1] )
            else:
                try:
                    Hfun = Hessian(rm.get_total_log_likelihood)
                    H = Hfun(m_opt_tmp)
                except:
                    print( 'Error in differentiation' )
                else:
                    try:
                        L = np.linalg.cholesky( -H )
                    except:
                        print( 'Hessian not positive-definite' )
                        count_errors += 1
                    else:
                        count_modes += 1
                        count_runs += 1
                        if f_opt < f_opt_tmp and not np.allclose(f_opt, f_opt_tmp):
                            count_modes = 1
                            f_opt = f_opt_tmp
                            m_opt[:] = m_opt_tmp
                            s_opt = np.linalg.inv(L)
                            s_opt = s_opt.T.dot(s_opt)
    		
            if count_errors >= 5:
                break
            else:                
                if count_runs >= 100 or count_modes == 10:
                    break                    
                            
                        
        return m_opt, s_opt, f_opt

        
        
        
        