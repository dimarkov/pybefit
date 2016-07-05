# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:13:45 2016

@author: Dimitrije Markovic
"""

class ResponseModel(object):
    """Base class for response models"""
    def __init__(self, responses, stimuli, prior, **kwargs):
        self.responses = responses #set the messured behavioral responses
        self.stimuli = stimuli #set the presented stimuli
        self.prior = prior #set prior parameter probability
        
class SimulatedRM(ResponseModel):
    
    def getSimulatedLogLikelihood(self, pvals, ind):
        """Method for estimating marginal log-likelihood"""
        pass
    
    def distance(self, simresponses, ind):
        """
        Estimate the distance of simulated responses from the 
        subjects response at trial ind
        """
        pass
    
class AnalyticalRM(ResponseModel):
    
    def getTotalLogLikelihood(self, pvals):
        """Return the sum of response log-likelihoods over all trials"""
        pass
    
    def getLogLikelihood(self, ind):
        """Return the response log-likelihood at trial ind"""
	pass
