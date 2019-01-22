"""This module contains the non-hierachical implementaion of the 
parametric model.
"""

import pandas as pd

import torch
from torch import zeros, ones
from torch.distributions import constraints

import pyro.distributions as dist
from pyro import sample, param, plate

from .infer import Inferrer

__all__ = [
    'Normal'        
]

class Normal(Inferrer):
    
    def __init__(self, agent, stimulus, responses, mask = None, fixed_params = None):
        super(Normal, self).__init__(agent, stimulus, responses, mask=mask, fixed_params=fixed_params)

    def model(self):
        """
        Generative model of behavior with flat (normal) prior over free model parameters.
        """
        nsub = self.runs #number of subjects
        npar = self.npars #number of parameters

        # define prior mean over model parametrs and subjects
        with plate('subjects', nsub):
                locs = sample('locs', dist.Normal(zeros(npar), 20.*ones(npar)).to_event(1))
                
        if self.fixed_values:
            x = zeros(nsub, self.agent.npars)
            x[:, self.locs['fixed']] = self.values
            x[:, self.locs['free']] = locs
        else:
            x = locs

        self.agent.set_parameters(x)
        
        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                states = self.stimulus['states'][:,b,t]
                offers = self.stimulus['offers'][:,b,t]
                
                self.agent.update_beliefs(b, t, states, offers)
                self.agent.planning(b,t)
        
        logits = self.agent.logprobs[self.notnans]        
        responses = self.rdata[self.notnans]
        
        with plate('observations', responses.shape[0]):
            sample('obs', dist.Categorical(logits=logits), obs=responses)
            
    def guide(self):
        """Approximate posterior for the horseshoe prior. We assume posterior in the form 
        of the multivariate normal distriburtion for the global mean and standard deviation
        and multivariate normal distribution for the parameters of each subject independently.
        """
        nsub = self.runs #number of subjects
        npar = self.npars #number of parameters
        
        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('scale_tril_locs', torch.eye(npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)

        with plate('subjects', nsub):
            locs = sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
        
        return {'locs': locs}
            
    def sample_posterior(self, labels, n_samples=10000):
        """Generate samples from posterior distribution. Only valid after 
        infer_posterior has finished execution.
        """
        nsub = self.runs
        npar = self.npars
        assert npar == len(labels)
        
        trans_pars = zeros(n_samples, nsub, npar)
        
        for i in range(n_samples):
            trans_pars[i] = self.guide()['locs'].detach()
            
        subject_label = torch.arange(1, nsub+1).repeat(n_samples, 1).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npar).numpy(), columns=labels)
        tp_df['subject'] = subject_label.numpy()
        
        return tp_df
    
    def _get_quantiles(self, quantiles):
        """
        Returns posterior quantiles each latent variable. 

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        
        quantiles = torch.tensor(quantiles).reshape(1, 3)

        
        m_locs = param('m_locs').reshape(-1, 1)
        s_locs = param('scale_tril_locs').diagonal(dim1=-2, dim2=-1).reshape(-1, 1)
        
        latents = dist.Normal(m_locs, s_locs).icdf(quantiles).reshape(self.runs, -1, 3)
        result = {'locs': latents}

        return result