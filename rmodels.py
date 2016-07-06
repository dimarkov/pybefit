# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:13:45 2016

@author: Dimitrije Markovic
"""

class ResponseModel(object):
    """Base class for response models"""
    def __init__(self, stimuli, prior, perceptual_model, responses=None, params = None, **kwargs):
        self.stimuli = stimuli #set the presented stimuli
        self.prior = prior #set prior parameter probability
	self.pm = perceptual_model #set the perceptual model
	
	#generate the behavioral responses or use the measured behavioral responses
	if(responses == None):
		if(params == None):
			print "Error: responses or response model parameters have not been set"
		else:
			self.params = params
			self.responses = self.generateResponses()
	else:
		self.responses = responses
	
    def generateResponses(self):
	"""Generate behavioral responses"""

        
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
