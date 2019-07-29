"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch import zeros, ones, tensor

from pyro import clear_param_store, get_param_store
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.enum import get_importance_trace
from pyro.optim import Adam

__all__ = [
    'Inferrer'        
]

class Inferrer(object):
    
    def __init__(self, agent, stimulus, responses, mask=None, fixed_params=None):
        
        self.agent = agent # agent used for computing response probabilities
        self.stimulus = stimulus # stimulus and action outcomes presented to each participant
        self.responses = responses # measured behavioral data accross all subjects
        self.nb, self.nt, self.runs = self.responses.shape
        
        # set a mask for excluding certain responses (e.g. NaN resonses) from 
        # the computations of the log-model evidence and the posterior beliefs over
        # parameter values
        if mask is not None:
            self.notnans = mask
        else:
            self.notnans = ones(self.nb, self.nt, self.runs, dtype=torch.uint8)
            
        if fixed_params is not None:
            n_fixed = len(fixed_params['labels'])
            self.npar = agent.npar - n_fixed
            
            self.locs = {}
            self.locs['fixed'] = fixed_params['labels']
            self.locs['free'] = list(set(range(agent.npar)) - \
                                set(fixed_params['labels']))
            self.values = fixed_params['values']
            self.fixed_values = True
        else:
            self.npar = agent.npar
            self.fixed_values = False

    def model(self):
        """
        Full generative model of behavior.
        """
        raise NotImplementedError
            
    def guide(self):
        """Approximate posterior over model parameters.
        """
        raise NotImplementedError

    def infer_posterior(self, 
                        iter_steps = 1000,
                        num_particles = 10,
                        optim_kwargs = {'lr':.1}):
        """Perform stochastic variational inference over free model parameters.
        """

        clear_param_store()

        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=Trace_ELBO(num_particles=num_particles, 
                                  vectorize_particles=False))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % tensor(loss[-20:]).mean())
                
        self.loss = loss

    def sample_posterior(self, labels, n_samples=10000):
        """Sample from the posterior distribution over model parameters.
        """
        raise NotImplementedError

    
    def _get_quantiles(self, quantiles):
        """
        Returns posterior quantiles each latent variable. Example::

            print(agent.get_quantiles([0.05, 0.5, 0.95]))

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        
        raise NotImplementedError
    
    def formated_results(self, par_names, labels=None):
        """Returns median, 5th and 95th percentile for each parameter and subject. 
        """
        nsub = self.runs
        npar = self.npar
        
        if labels is None:
            labels = par_names
        
        
        quantiles = self._get_quantiles([.05, .5, .95])
            
        locs = quantiles['locs'].transpose(dim0=0, dim1=-1).transpose(dim0=1, dim1=-1)

        if self.fixed_values:
            x = zeros(3, nsub, npar)
            x[..., self.locs['fixed']] = self.values
            x[..., self.locs['free']] = locs.detach()
        else:
            x = locs.detach()
        
        self.agent.set_parameters(x, set_variables=False)
        
        par_values = {}
        for name in par_names:
            values = getattr(self.agent, name)
            if values.dim() < 3:
                values = values.unsqueeze(dim=-1)
            par_values[name] = values
        
        count = {}
        percentiles = {}
        for name in par_names:
            count.setdefault(name, 0)
            for lbl in labels:
                if lbl.startswith(name):
                    percentiles[lbl] = par_values[name][..., count[name]].numpy().reshape(-1)
                    count[name] += 1
        
        df_percentiles = pd.DataFrame(percentiles)
        
        subjects = torch.arange(1, nsub+1).repeat(3, 1).reshape(-1)
        df_percentiles['subjects'] = subjects.numpy()
        
        from numpy import tile, array
        variables = tile(array(['5th', 'median', '95th']), [nsub, 1]).T.reshape(-1)
        df_percentiles['variables'] = variables
        
        return df_percentiles.melt(id_vars=['subjects', 'variables'], var_name='parameter')

    def get_log_evidence_per_subject(self, num_particles = 100, max_plate_nesting=1):
        """Return subject specific log model evidence"""
        
        model = self.model
        guide = self.guide
        notnans = self.notnans
        
        elbo = zeros(self.runs)
        for i in range(num_particles):
            model_trace, guide_trace = get_importance_trace('flat', max_plate_nesting, model, guide)
            obs_log_probs = zeros(notnans.shape)
            for site in model_trace.nodes.values():
                if site['name'].startswith('obs'):
                    b, t = np.array(site['name'].split('_')[-2:]).astype(int)
                    obs_log_probs[b, t, notnans[b, t]] = site['log_prob'].detach()
                elif site['name'] == 'locs':
                    elbo += site['log_prob'].detach()
            elbo += torch.einsum('ijk->k', obs_log_probs)

            for site in guide_trace.nodes.values():
                if site['name'] == 'locs':
                    elbo -= site['log_prob'].detach()
        
        return elbo/num_particles