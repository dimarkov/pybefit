# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:05:58 2016

@author: Dimitrije Markovic
"""

import optmethods
from numpy as np
from numdifftools import Hessian

class Inference(object):
    """Bese class for various inference methods"""
    def __init__(self, **kwargs):
        pass
    
    def inferPosterior(self, model, options):
        pass

class LAinference(Inference):
    
    def __init__(self, method = 'cmaes', ftol = 1e-8, xtol = 1e-12, bound = 20, **kwargs):
        #set optimization method; possible choices are 'cmaes' and 'isres'
        self.method = method
        
        #set search bounds
        self.bounds = {'lb': -bound, 'ub': bound}
        if(kwargs.has_key('bounds')):
            self.bounds = {'lb': kwargs['bounds'][0,:],
                           'ub': kwargs['bounds'][1,:]}
                
        #set function tolerance
        self.ftol = ftol
        
        #set parameter value tolerance
        self.xtol = xtol
        
    def inferPosterior(self, rmodel, options):
        
        f_opt = -np.inf #model log-evidence
        m_opt = np.zeros(rmodel.npars) #posterior expectation
        s_opt = np.inf*np.eye(rmodel.npars) #posterior covariance matrix
        
        x_init = self.bounds['lb'] + self.bounds['ub']
        opt_method = getattr(optmethods, self.method) #optimization method
        
        f_opt, m_opt, stopflag = opt_method(rmodel.getTotalLogLikelihood, 
                                                     rmodel.npars, 
                                                     self.ftol, self.xtol, 
                                                     self.bounds, x_init)
                                                     
        try:
            Hfun = Hessian(rmodel.getTotalLogLikelihood)
            H = Hfun(m_opt) #compute the hessian around the mode
            L = np.linalg.cholesky(H) #check if positive definite
        except:
            print 'Hessian at the found mode could not be computed'
        else:
            f_opt = f_opt - np.log(np.linalg.det(L))
            s_opt = np.linalg.inv(L)
            s_opt = np.dot(s_opt.T, s_opt)
            
        return f_opt, m_opt, s_opt, stopflag