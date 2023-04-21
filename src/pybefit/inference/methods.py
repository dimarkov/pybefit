"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
import pyro
import pyro.infer as pinfer

import numpyro.infer as ninfer

import jax.random as jr

import optax
import numpy as np
import pandas as pd

from numpyro.optim import optax_to_numpyro

from multipledispatch import dispatch

from tqdm.auto import trange

from pyro import clear_param_store, get_param_store

from pyro.optim import Adam

from .models import NumpyroModel, NumpyroGuide, PyroModel, PyroGuide

adabelief = lambda *args, **kwargs: optax.adabelief(*args, eps=1e-8, eps_root=1e-8, **kwargs)

__all__ = [
    'run_svi',
    'run_nuts',
    'format_posterior_samples'
]

default_dict_pyro_svi = dict(
    enumerate=False,            
    iter_steps=10000,
    optim_kwargs={'lr': 1e-3},
    elbo_kwargs=dict(
        num_particles=10,
        vectorize_particles=True
    ),
    svi_kwargs=dict(
        progress_bar=True,
        stable_update=True
    ),
    sample_kwargs=dict(
        num_samples=100
    )
)

# Pyro/Pytorch based models
@dispatch(PyroModel, PyroGuide, dict, opts=dict)
def run_svi(model,
            guide,
            data,
            *,
            opts=default_dict_pyro_svi
        ):
    """Perform SVI over free model parameters.
    """

    clear_param_store()

    if opts['enumerate']:
        elbo = pinfer.TraceEnum_ELBO(**opts['elbo_kwargs'])
    else:
        elbo = pinfer.Trace_ELBO(**opts['elbo_kwargs'])
    
    svi = pinfer.SVI(
        model=model,
        guide=guide,
        optim=Adam(opts['optim_kwargs']),
        loss=elbo
    )

    loss = []
    pbar = trange(opts['iter_steps'], position=0)
    for _ in pbar:
        loss.append(svi.step(data))
        pbar.set_description("Mean ELBO %6.2f" % np.mean(loss[-20:]))
        if np.isnan(loss[-1]):
            break
            print('loss returned NAN value')

    param_store = get_param_store()
    params = {}
    for name in param_store.get_all_param_names():
        params[name] = pyro.param(name)

    results = {
        'loss': loss,
        'params': params
    }

    pred = pinfer.Predictive(model, guide=guide, **opts['sample_kwargs'])
    samples = pred(data)

    return samples, svi, results


@dispatch(PyroModel, opts=dict)
def run_nuts(
    model,
    *,
    opts=dict(
        num_samples=1000,            
        num_warmup=100,
        sampler_kwargs={'kernel': {}, 'mcmc': {}}
    )
):
    """Perform SVI over free model parameters.
    """

    clear_param_store()

    kernel = pinfer.NUTS(model, **opts['sampler_kwargs']['kernel'])
    mcmc = pinfer.MCMC(
        kernel, 
        opts['num_samples'], 
        warmup_steps=opts['num_warmup'],
        **opts['sampler_kwargs']['mcmc']
    )

    mcmc.run()
    samples = mcmc.get_samples()

    return samples, mcmc


default_dict_numpyro_svi = dict(
    seed=0,
    enumerate=False,            
    iter_steps=10000,
    optim=None,
    optim_kwargs={'learning_rate': 1e-3},
    elbo_kwargs=dict(
        num_particles=10,
        max_plate_nesting=1,
    ),
    svi_kwargs=dict(
        progress_bar=True,
        stable_update=True
    ),
    sample_kwargs=dict(
        num_samples=100
    )
)

# Numpyro/Jax based models
@dispatch(NumpyroModel, NumpyroGuide, dict, opts=dict)
def run_svi(model,
            guide,
            data,
            *,
            opts=default_dict_numpyro_svi
        ):
    """Estimate posterior over free model parameters using stochastic variational inference.
    """
    rng_key = jr.PRNGKey(opts['seed'])

    if opts['enumerate']:
        elbo = ninfer.TraceEnum_ELBO(**opts['elbo_kwargs'])
    else:
        elbo = ninfer.TraceGraph_ELBO(num_particles=opts['elbo_kwargs']['num_particles'])
    
    if opts['optim'] is None:
        optim = optax_to_numpyro(adabelief(**opts['optim_kwargs']))
    else:
        optim = optim
    
    svi = ninfer.SVI(model, guide, optim, elbo)
    rng_key, _rng_key = jr.split(rng_key)
    results = svi.run(
        _rng_key,
        opts['iter_steps'],
        progress_bar=opts['svi_kwargs']['progress_bar'],
        stable_update=opts['svi_kwargs']['stable_update'],
        init_state = opts['svi_kwargs'].pop('init_state', None),
        data=data
    )

    rng_key, _rng_key = jr.split(rng_key)
    pred = ninfer.Predictive(model, guide=guide, params=results.params, **opts['sample_kwargs'])
    samples = pred(_rng_key, data=data)

    return samples, svi, results

default_dict_numpyro_nuts = dict(
    seed=0,
    num_samples=1000,            
    num_warmup=100,
    sampler_kwargs={
        'kernel': {}, 
        'mcmc': dict(progress_bar=True)
    }
)

@dispatch(NumpyroModel, dict, opts=dict)
def run_nuts(
    model,
    data,
    *,
    opts=default_dict_numpyro_nuts
):
    """Estimate posterior over free model parameters using No-U-Turn sampler.
    """
    rng_key = jr.PRNGKey(opts['seed'])

    kernel = ninfer.NUTS(model, **opts['sampler_kwargs']['kernel'])
    mcmc = ninfer.MCMC(
        kernel,
        num_warmup=opts['num_warmup'],
        num_samples=opts['num_samples'], 
        **opts['sampler_kwargs']['mcmc']
    )

    rng_key, _rng_key = jr.split(rng_key)
    mcmc.run(_rng_key, data=data)
    samples = mcmc.get_samples()

    return samples, mcmc

def format_posterior_samples(labels, samples, transform, *args, **kwargs):
    """Format samples into DataFrames based on agent transform
    """

    num_samples, num_agents, num_params = samples['z'].shape
    assert num_params == len(labels)

    subject_id = np.arange(1,  num_agents + 1)[None].repeat(num_samples, axis=0).reshape(-1)

    agent = transform(samples['z'], *args, **kwargs)

    pars = []
    for lab in labels:
        pars.append(np.array(getattr(agent, lab).reshape(-1)))
        
    pars_df = pd.DataFrame(data=np.stack(pars, -1), columns=labels)
    pars_df['subject'] = subject_id

    trans_pars_df = pd.DataFrame(data=samples['z'].reshape(-1, num_params), columns=labels)
    trans_pars_df['subject'] = subject_id

    return trans_pars_df, pars_df


#     def sample_posterior_marginal(self, n_samples=100):
#         '''Draw posterior samples over descrete variables in the model'''
#         if self.enumerate:
#             elbo = TraceEnum_ELBO()
#             post_discrete_samples = {}

#             pbar = tqdm(range(n_samples), position=0)

#             for n in pbar:
#                 pbar.set_description("Sample posterior discrete marginal")
#                 # get marginal posterior over planning depths
#                 post_sample = elbo.compute_marginals(self.model, self.guide)
#                 for name in post_sample.keys():
#                     post_discrete_samples.setdefault(name, [])
#                     post_discrete_samples[name].append(post_sample[name].probs.detach().clone())

#             for name in post_discrete_samples.keys():
#                 post_discrete_samples[name] = torch.stack(post_discrete_samples[name]).numpy()

#             return post_discrete_samples
#         else:
#             print("No enumerate variables in the model")
#             return -1

#     def _get_quantiles(self, quantiles):
#         """
#         Returns posterior quantiles each latent variable. Example::

#             print(agent.get_quantiles([0.05, 0.5, 0.95]))

#         :param quantiles: A list of requested quantiles between 0 and 1.
#         :type quantiles: torch.tensor or list
#         :return: A dict mapping sample site name to a list of quantile values.
#         :rtype: dict
#         """

#         raise NotImplementedError

#     def formated_results(self, par_names, labels=None):
#         """Returns median, 5th and 95th percentile for each parameter and subject.
#         """
#         nsub = self.runs
#         npar = self.npar

#         if labels is None:
#             labels = par_names

#         quantiles = self._get_quantiles([.05, .5, .95])

#         locs = quantiles['locs'].transpose(dim0=0, dim1=-1).transpose(dim0=1, dim1=-1)

#         if self.fixed_values:
#             x = zeros(3, nsub, npar)
#             x[..., self.locs['fixed']] = self.values
#             x[..., self.locs['free']] = locs.detach()
#         else:
#             x = locs.detach()

#         self.agent.set_parameters(x, set_variables=False)

#         par_values = {}
#         for name in par_names:
#             values = getattr(self.agent, name)
#             if values.dim() < 3:
#                 values = values.unsqueeze(dim=-1)
#             par_values[name] = values

#         count = {}
#         percentiles = {}
#         for name in par_names:
#             count.setdefault(name, 0)
#             for lbl in labels:
#                 if lbl.startswith(name):
#                     percentiles[lbl] = par_values[name][..., count[name]].numpy().reshape(-1)
#                     count[name] += 1

#         df_percentiles = pd.DataFrame(percentiles)

#         subjects = torch.arange(1, nsub+1).repeat(3, 1).reshape(-1)
#         df_percentiles['subjects'] = subjects.numpy()

#         from numpy import tile, array
#         variables = tile(array(['5th', 'median', '95th']), [nsub, 1]).T.reshape(-1)
#         df_percentiles['variables'] = variables

#         return df_percentiles.melt(id_vars=['subjects', 'variables'], var_name='parameter')

import torch

def get_log_evidence_per_subject(svi, num_agents, *args, num_particles=100, max_plate_nesting=1, **kwargs):
    """Return subject specific log model evidence"""

    model = svi.model
    guide = svi.guide

    elbo = torch.zeros(num_agents)
    for i in range(num_particles):
        model_trace, guide_trace = pyro.infer.enum.get_importance_trace('flat', max_plate_nesting, model, guide, args, kwargs)
        for site in model_trace.nodes.values():
            if site['name'].startswith('obs'):
                elbo += site['log_prob'].detach()
            elif site['name'] == 'locs':
                elbo += site['log_prob'].detach()

        for site in guide_trace.nodes.values():
            if site['name'] == 'locs':
                elbo -= site['log_prob'].detach()

    return elbo/num_particles
