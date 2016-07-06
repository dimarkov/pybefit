# -*- coding: utf-8 -*-
"""
Created on Tue July 6 14:13:45 2016

@author: Dimitrije Markovic
"""

class PerceptualModel(object):
    """Base class for perceptual models"""
    def __init__(self, stimuli, prior, params = None, **kwargs):
        self.stimuli = stimuli #set the presented stimuli
        self.prior = prior #set prior parameter probability
	
	if(params == None):
		self.params = self.sampleFromPrior()
	else:
		self.params = params

	
    def sampleFromPrior(self):
	"""
	Sample parameter values of the perceptual model from the prior distribution.
	"""

class AnalyticalPM(PerceptualModel):
    
    def inferPosteriorBeliefs(self):
        """Estimate the posterior beliefs over an experimental block """
        pass
    
    def updateBeliefs(self, ind):
        """Estimate the updated beliefs after observing a stimulus at trial ind"""
	pass
