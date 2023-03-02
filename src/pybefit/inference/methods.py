"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
import pyro
import pyro.infer as pinfer

import numpyro.infer as ninfer

import jax.random as jr

import optax

from numpyro.optim import optax_to_numpyro

from multipledispatch import dispatch
from typing import Dict, Any

from tqdm.auto import tqdm

from pyro import clear_param_store, get_param_store

from pyro.optim import Adam

from .numpyro.models import NumpyroModel, NumpyroGuide
from .pyro.models import PyroModel, PyroGuide

adabelief = lambda *args, **kwargs: optax.adabelief(*args, eps=1e-8, eps_root=1e-8, **kwargs)

__all__ = [
    'run_svi',
    'run_nuts'
]

# Pyro/Pytorch based models
@dispatch(PyroModel, PyroGuide, opts=dict)
def run_svi(model,
            guide,
            *,
            opts=dict(
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
        ):
    """Perform SVI over free model parameters.
    """

    clear_param_store()

    if enumerate:
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
    pbar = tqdm(range(opts['iter_steps']), position=0)
    for _ in pbar:
        loss.append(svi.step())
        pbar.set_description("Mean ELBO %6.2f" % np.mean(loss[-20:]))
        if np.isnan(loss[-1]):
            break

    param_store = get_param_store()
    params = {}
    for name in param_store.get_all_param_names():
        params[name] = pyro.param(name)

    results = {
        'loss': loss,
        'params': params
    }

    pred = pinfer.Predictive(model, guide=guide, **opts['sample_kwargs'])
    samples = pred()

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
        data=data,
        **opts['svi_kwargs']
    )

    rng_key, _rng_key = jr.split(opts['rng_key'])
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

# class Inferrer(object):

#     def __init__(self, agent, stimulus, responses, mask=None, fixed_params=None, enumerate=False):

#         self.agent = agent  # agent used for computing response probabilities
#         self.stimulus = stimulus  # stimulus and action outcomes presented to each participant
#         self.responses = responses  # measured behavioral data accross all subjects
#         self.nb, self.nt, self.runs = self.responses.shape

#         # set a mask for excluding certain responses (e.g. NaN resonses) from
#         # the computations of the log-model evidence and the posterior beliefs over
#         # parameter values
#         if mask is not None:
#             self.notnans = mask
#         else:
#             self.notnans = ones(self.nb, self.nt, self.runs, dtype=torch.bool)

#         if fixed_params is not None:
#             n_fixed = len(fixed_params['labels'])
#             self.npar = agent.npar - n_fixed

#             self.locs = {}
#             self.locs['fixed'] = fixed_params['labels']
#             self.locs['free'] = list(set(range(agent.npar)) - set(fixed_params['labels']))
#             self.values = fixed_params['values']
#             self.fixed_values = True
#         else:
#             self.npar = agent.npar
#             self.fixed_values = False

#         self.enumerate = enumerate

#     def model(self):
#         """
#         Full generative model of behavior.
#         """
#         raise NotImplementedError

#     def guide(self):
#         """Approximate posterior over model parameters.
#         """
#         raise NotImplementedError

#     def infer_posterior(self,
#                         iter_steps=10000,
#                         num_particles=100,
#                         optim_kwargs={'lr': .01}):
#         """Perform SVI over free model parameters.
#         """

#         clear_param_store()

#         model = self.model
#         guide = self.guide
#         if self.enumerate:
#             elbo = TraceEnum_ELBO(num_particles=num_particles, vectorize_particles=True)
#         else:
#             elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
#         svi = SVI(model=model,
#                   guide=guide,
#                   optim=Adam(optim_kwargs),
#                   loss=elbo)

#         loss = []
#         pbar = tqdm(range(iter_steps), position=0)
#         for step in pbar:
#             loss.append(svi.step())
#             pbar.set_description("Mean ELBO %6.2f" % np.mean(loss[-20:]))
#             if np.isnan(loss[-1]):
#                 break

#         param_store = get_param_store()
#         parameters = {}
#         for name in param_store.get_all_param_names():
#             parameters[name] = param(name)

#         self.parameters = parameters
#         self.loss = loss
#         self.elbo = elbo

#     def sample_posterior(self, labels, num_samples=1000):
#         """Generate samples from posterior distribution.
#         """
#         nsub = self.runs
#         npar = self.npar
#         assert npar == len(labels)

#         sites = ['sigma', 'mu', 'locs']
#         predict = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=sites, parallel=True)
#         samples = predict()

#         subject_id = torch.arange(1, nsub+1).repeat(num_samples, 1).reshape(-1)
#         trans_pars_df = pd.DataFrame(data=samples['locs'].detach().reshape(-1, npar).numpy(), columns=labels)
#         trans_pars_df['subject'] = subject_id.numpy()

#         locs = samples['locs'].detach()
#         if self.fixed_values:
#             x = zeros(locs.shape[:-1] + (self.agent.npar,))
#             x[..., self.locs['fixed']] = self.values
#             x[..., self.locs['free']] = locs
#         else:
#             x = locs

#         self.agent.set_parameters(x)
#         pars = []
#         for lab in labels:
#             pars.append(getattr(self.agent, lab[2:-1]).reshape(-1).numpy())
#         pars_df = pd.DataFrame(data=np.stack(pars, -1), columns=labels)
#         pars_df['subject'] = subject_id.numpy()

#         mu = np.take(samples['mu'].detach().numpy(), 0, axis=-2)
#         sigma = np.take(samples['sigma'].detach().numpy(), 0, axis=-2)

#         mu_df = pd.DataFrame(data=mu, columns=labels)
#         sigma_df = pd.DataFrame(data=sigma, columns=labels)

#         return (trans_pars_df, pars_df, mu_df, sigma_df)


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

#     def get_log_evidence_per_subject(self, *args, num_particles=100, max_plate_nesting=1, **kwargs):
#         """Return subject specific log model evidence"""

#         model = self.model
#         guide = self.guide

#         elbo = zeros(self.runs)
#         for i in range(num_particles):
#             model_trace, guide_trace = get_importance_trace('flat', max_plate_nesting, model, guide, args, kwargs)
#             for site in model_trace.nodes.values():
#                 if site['name'].startswith('obs'):
#                     elbo += site['log_prob'].detach()
#                 elif site['name'] == 'locs':
#                     elbo += site['log_prob'].detach()

#             for site in guide_trace.nodes.values():
#                 if site['name'] == 'locs':
#                     elbo -= site['log_prob'].detach()

#         return elbo/num_particles
