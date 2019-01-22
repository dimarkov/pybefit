"""This module contains the hierarchical implementations of the parametric model.
"""

import pandas as pd

import torch
from torch import zeros, ones
from torch.distributions import constraints, biject_to

import pyro.distributions as dist
from pyro import sample, param, plate
from pyro.distributions.util import sum_rightmost

from .infer import Inferrer

__all__ = [
    'Horseshoe'        
]

class Horseshoe(Inferrer):
    
    def __init__(self, agent, stimulus, responses, mask=None, fixed_params=None):
        super(Horseshoe, self).__init__(agent, stimulus, responses, mask=mask, fixed_params=fixed_params)

    def model(self):
        """
        Generative model of behavior with a hierarchical (horseshoe)
        prior over free model parameters.
        """
        runs = self.runs  # number of independent runs of experiment
        npar = self.npar  # number of parameters

        # define hyper priors over model parameters.
        # each model parameter has a hyperpriors defining group level mean
        mu = sample('mu', dist.Normal(zeros(npar), 20.*ones(npar)).to_event(1))

        # define prior uncertanty over model parameters and subjects
        sigma = sample('sigma', dist.HalfCauchy(1.).expand([npar]).to_event(1))

        # define prior mean over model parametrs and subjects
        with plate('runs', runs):
                locs = sample('locs', dist.Normal(mu, sigma).to_event(1))
        

        if self.fixed_values:
            x = zeros(runs, self.agent.npars)
            x[:, self.locs['fixed']] = self.values
            x[:, self.locs['free']] = locs
        else:
            x = locs

        self.agent.set_parameters(x)
        
        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                outcomes = self.stimulus[b, t]
                
                self.agent.update_beliefs(b, t, **outcomes)
                self.agent.planning(b, t)
        
        logprobs = torch.stack(self.agent.logprobs).reshape(self.nb, self.nt, -1)[self.notnans]
        responses = self.responses[self.notnans]
        
        with plate('observations', responses.shape[0]):
            sample('obs', dist.Bernoulli(logits=logprobs), obs=responses)
            
    def guide(self):
        """Approximate posterior for the horseshoe prior. We assume posterior in the form 
        of the multivariate normal distriburtion for the global mean and standard deviation
        and multivariate normal distribution for the parameters of each subject independently.
        """
        nsub = self.runs #number of subjects
        npar = self.npar #number of parameters
        trns = biject_to(constraints.positive)

        
        m_hyp = param('m_hyp', zeros(2*npar))
        st_hyp = param('scale_tril_hyp', 
                              torch.eye(2*npar), 
                              constraint=constraints.lower_cholesky)
        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, 
                                                  scale_tril=st_hyp), 
                            infer={'is_auxiliary': True})
        
        unc_mu = hyp[:npar]
        unc_sigma = hyp[npar:]
    
        c_sigma = trns(unc_sigma)
    
        ld_sigma = trns.inv.log_abs_det_jacobian(c_sigma, unc_sigma)
        ld_sigma = sum_rightmost(ld_sigma, ld_sigma.dim() - c_sigma.dim() + 1)
    
        mu = sample("mu", dist.Delta(unc_mu, event_dim=1))
        sigma = sample("sigma", dist.Delta(c_sigma, log_density=ld_sigma, event_dim=1))
        
        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('scale_tril_locs', torch.eye(npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)

        with plate('subjects', nsub):
            locs = sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
        
        return {'mu': mu, 'sigma': sigma, 'locs': locs}
            
    def sample_posterior(self, labels, n_samples=10000):
        """Generate samples from posterior distribution.
        """
        nsub = self.runs
        npar = self.npar
        assert npar == len(labels)
        
        keys = ['locs', 'sigma', 'mu']
        
        trans_pars = zeros(n_samples, nsub, npar)
        
        mu_group = zeros(n_samples, npar)
        sigma_group = zeros(n_samples, npar)
        
        for i in range(n_samples):
            sample = self.guide()
            for key in keys:
                sample.setdefault(key, ones(1))
                
            mu = sample['mu']
            sigma = sample['sigma']
            locs = sample['locs']
            
            trans_pars[i] = locs.detach()
            
            mu_group[i] = mu.detach()
            sigma_group[i] = sigma.detach()
        
        subject_label = torch.arange(1, nsub+1).repeat(n_samples, 1).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npar).numpy(), columns=labels)
        tp_df['subject'] = subject_label.numpy()
        
        g_df = pd.DataFrame(data=mu_group.numpy(), columns=labels)
        sg_df = pd.DataFrame(data=sigma_group.numpy(), columns=labels)
        
        return (tp_df, g_df, sg_df)
    
    def _get_quantiles(self, quantiles):
        """
        Returns posterior quantiles each latent variable.

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        
        self.means = [param('m_locs'), param('m_hyp')]
        self.stds = [param('scale_tril_hyp'), param('scale_tril_locs')]
        
        quantiles = torch.tensor(quantiles).reshape(1, 3)

        
        m_locs = param('m_locs').reshape(-1, 1)
        s_locs = param('scale_tril_locs').diagonal(dim1=-2, dim2=-1).reshape(-1, 1)
        
        latents = dist.Normal(m_locs, s_locs).icdf(quantiles).reshape(self.runs, -1, 3)
        result = {'locs': latents}

        m_hyp = param('m_hyp').reshape(-1, 1)
        s_hyp = param('scale_tril_hyp').diagonal(dim1=-2, dim2=-1).reshape(-1, 1)
        
        latents = dist.Normal(m_hyp, s_hyp).icdf(quantiles).reshape(-1, 1)
        
        result['mu'] = latents[:self.npar]
        result['sigma'] = latents[self.npar:].exp()
                
        return result