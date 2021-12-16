"""This module contains the hierarchical implementations of the parametric model.
"""
import torch
import numpy as np
from tqdm import tqdm

from torch import zeros, ones, tensor
from torch.distributions import constraints, biject_to

import pyro.distributions as dist
from pyro import sample, param, plate, poutine
from pyro.distributions.util import sum_rightmost
from pyro.ops.indexing import Vindex

from pyro import clear_param_store
from pyro.infer import SVI, TraceEnum_ELBO, infer_discrete
from pyro.optim import Adam

from .infer import Inferrer

__all__ = [
    'DirichletProcessPrior'
]

from torch.nn.functional import pad


def stick_breaking(x):
    return pad(x, (0, 1), value=1) * pad((1 - x).cumprod(-1), (1, 0), value=1)


class DirichletProcessPrior(Inferrer):

    def __init__(self, agent, stimulus, responses, mask=None, fixed_params=None, max_components=5):
        self.kmax = max_components
        super(DirichletProcessPrior, self).__init__(agent, stimulus, responses, mask=mask, fixed_params=fixed_params)

    def model(self):
        """
        Generative model of behavior with a hierarchical (horseshoe)
        prior over free model parameters.
        """
        runs = self.runs  # number of independent runs of experiment
        npar = self.npar  # number of parameters
        kmax = self.kmax  # maximal number of components in the mixture model

        # parametrise gamma with alpha and rho=alpha/beta
        maa = param('ma_a', ones(1), constraint=constraints.positive)
        mar = param('ma_r', ones(1), constraint=constraints.positive)
        alpha = sample('alpha', dist.Gamma(maa, maa/mar))

        with plate('components', kmax-1):
            beta = sample('beta', dist.Beta(1, alpha))

        # define hyper priors over model parameters
        with plate('classes', kmax, dim=-2):
            a = param('a', ones(kmax, npar), constraint=constraints.positive)
            lam = param('lam', ones(kmax, npar), constraint=constraints.positive)
            dg = dist.Gamma(a, a/lam).to_event(1)
            tau = sample('tau', dg)

            sig = 1/torch.sqrt(tau)

            m = param('m', zeros(kmax, npar))
            s = param('s', ones(kmax, npar), constraint=constraints.positive)
            mu = sample('mu', dist.Normal(m, s*sig).to_event(1))

        print(mu.shape, tau.shape)
        # define prior mean over model parameters and subjects
        with plate('runs', runs, dim=-2):
            d = sample('class', dist.Categorical(stick_breaking(beta)), infer={"enumerate": "parallel"})

            base_dist = dist.Normal(0., 1.).expand_by([npar])
            # transform = dist.transforms.AffineTransform(Vindex(mu.unsqueeze(-3))[..., d, :],
            #                                             Vindex(sig.unsqueeze(-3))[..., d, :])
            transform = dist.transforms.AffineTransform(mu[..., :1, :],
                                                        sig[..., :1, :])
            print(transform.loc.shape, base_dist.batch_shape, base_dist.event_shape)
            locs = sample('locs', dist.TransformedDistribution(base_dist, [transform]).to_event(1))

            if self.fixed_values:
                x = zeros(locs.shape[:-1] + (self.agent.npar,))
                x[..., self.locs['fixed']] = self.values
                x[..., self.locs['free']] = locs
            else:
                x = locs

            self.agent.set_parameters(x)

            for b in range(self.nb):
                for t in range(self.nt):
                    # update single trial
                    offers = self.stimulus['offers'][b, t]
                    self.agent.planning(b, t, offers)

                    outcomes = self.stimulus['outcomes'][b, t]
                    responses = self.responses[b, t]

                    mask = self.stimulus['mask'][b, t]
                    self.agent.update_beliefs(b, t, [responses, outcomes], mask=mask)

                    mask = self.notnans[b, t]
                    logits = self.agent.logits[-1]
                    sample('obs_{}_{}'.format(b, t),
                           dist.Categorical(logits=logits).mask(mask),
                           obs=responses)

    def guide(self):
        """Approximate posterior for the Dirichlet process prior.
        """
        nsub = self.runs  # number of subjects
        npar = self.npar  # number of parameters
        kmax = self.kmax  # maximum number of components

        gaa = param("ga_a", ones(1), constraint=constraints.positive)
        gar = param("ga_r", .1*ones(1), constraint=constraints.positive)
        sample('alpha', dist.Gamma(gaa, gaa/gar))

        gba = param("gb_beta_a", ones(kmax - 1), constraint=constraints.positive)
        gbb = param("gb_beta_b", ones(kmax - 1), constraint=constraints.positive)
        beta = sample("beta", dist.Beta(gba, gbb).to_event(1))

        with plate('classes', kmax, dim=-2):
            m_mu = param('m_mu', zeros(kmax, npar))
            st_mu = param('scale_tril_mu', torch.eye(npar).repeat(kmax, 1, 1), constraint=constraints.lower_cholesky)
            mu = sample("mu", dist.MultivariateNormal(m_mu, scale_tril=st_mu))

            m_tau = param('m_hyp', zeros(kmax, npar))
            st_tau = param('scale_tril_hyp', torch.eye(npar).repeat(kmax, 1, 1), constraint=constraints.lower_cholesky)
            mn = dist.MultivariateNormal(m_tau, scale_tril=st_tau)
            sample("tau", dist.TransformedDistribution(mn, [dist.transforms.ExpTransform()]))

        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('scale_tril_locs', torch.eye(npar).repeat(nsub, 1, 1), constraint=constraints.lower_cholesky)

        class_probs = param('class_probs', ones(nsub, kmax) / kmax, constraint=constraints.simplex)
        with plate('subjects', nsub, dim=-2):
            sample('class', dist.Categorical(class_probs), infer={"enumerate": "parallel"})
            sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

    def infer_posterior(self, iter_steps=10000, num_particles=100, optim_kwargs={'lr': .01}):
        """Perform SVI over free model parameters.
        """

        clear_param_store()

        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=TraceEnum_ELBO(num_particles=num_particles, vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % tensor(loss[-20:]).mean())
            if np.isnan(loss[-1]):
                break

        self.loss = loss

    def classifier(self, num_samples=1000, temperature=1):

        guide_trace = poutine.trace(self.guide).get_trace()
        trained_model = poutine.replay(self.model, trace=guide_trace)  # replay the globals

        classes = []
        for n in range(num_samples):
            # avoid conflict with data plate
            inferred_model = infer_discrete(trained_model, temperature=temperature, first_available_dim=-1)
            trace = poutine.trace(inferred_model).get_trace()
            classes.append(trace.nodes["class"]["value"])

        self.classes = torch.stack(classes)

        return self.classes

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

        return result
