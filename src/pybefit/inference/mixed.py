"""This module contains the hierarchical parametric model with 
descrete variables.
"""

import torch
from torch import zeros, ones
from torch.distributions import biject_to

import pyro.distributions as dist
from pyro.distributions import constraints
from pyro import sample, param, poutine, deterministic, plate, markov
from pyro.distributions.util import sum_rightmost
from pyro.ops.indexing import Vindex
from pyro.infer.reparam import LocScaleReparam

from .infer import Inferrer

__all__ = [
    'NormalGammaDiscreteDepth'
]


class NormalGammaDiscreteDepth(Inferrer):
    def __init__(self,
                 agent,
                 stimulus,
                 responses,
                 mask):

        super(NormalGammaDiscreteDepth, self).__init__(
            agent, 
            stimulus, 
            responses, 
            mask=mask, 
            fixed_params=None,
            enumerate=True)

        self.agent = agent
        self.mask = mask
        self.N = mask.sum(dim=0)

        self.subs = torch.arange(0, self.runs, 1, dtype=torch.long) # helper variable for indexing

        self.depth_transition = zeros(2, 3, 2, 3)
        self.depth_transition[0, :, 0] = torch.tensor([1., 0., 0.])
        self.depth_transition[0, :, 1] = torch.tensor([.5, .5, 0.])
        self.depth_transition[1] = torch.tensor([1., 0., 0.])

        self.states = stimulus['states']
        self.configs = stimulus['configs']
        self.conditions = stimulus['conditions']

    def model(self):
        agent = self.agent
        np = agent.npar  # number of parameters

        nblk = self.nb  # number of mini-blocks
        nsub = self.runs  # number of subjects

        # define prior uncertanty over model parameters and subjects
        a = param('a', ones(np), constraint=constraints.softplus_positive)
        lam = param('lam', ones(np), constraint=constraints.softplus_positive)
        var_tau = sample('var_tau', dist.Gamma(a, 1).to_event(1))

        sig = deterministic('sigma', torch.sqrt(lam/(a*var_tau)))

        # each model parameter has a hyperprior defining group level mean
        m = param('m', zeros(np))
        s = param('s', ones(np), constraint=constraints.softplus_positive)
        with poutine.reparam(config={"mu": LocScaleReparam()}):
            mu = sample("mu", dist.Normal(m, s*sig).to_event(1))

        ac12 = param("alphas12", ones(2, 2), constraint=constraints.softplus_positive)
        ac34 = param("alphas34", ones(2, 3), constraint=constraints.softplus_positive)

        with plate('subjects', nsub):
            with poutine.reparam(config={"locs": LocScaleReparam()}):
                locs = sample("locs", dist.Normal(mu, sig).to_event(1))
            
            # define priors over planning depth
            probs_c12 = sample("probs_cond_12", dist.Dirichlet(ac12).to_event(1))
            probs_c34 = sample("probs_cond_34", dist.Dirichlet(ac34).to_event(1))

        agent.set_parameters(locs)

        tmp = torch.nn.functional.pad(probs_c12, (0, 1), value=0.)
        priors = torch.concat([tmp, probs_c34], -2)

        for b in markov(range(nblk)):
            conditions = self.conditions[..., b]
            states = self.states[:, b]
            responses = self.responses[b]
            max_trials = conditions[-1] - 2
            noise = conditions[-2]

            tm = self.depth_transition[:, :, max_trials]
            for t in markov(range(3)):

                if t == 0:
                    res = None
                    probs = priors[..., self.subs, 2*max_trials + noise, :]
                else:
                    res = responses[t-1]
                    probs = tm[t-1, -1]

                agent.update_beliefs(b, t, states[:, t], conditions, res)
                agent.plan_actions(b, t)

                valid = self.mask[:, b, t]
                N = self.N[b, t]
                res = responses[t, valid]

                logits = agent.logits[-1][..., valid, :]
                probs = probs[..., valid, :]

                if t == 2:
                    agent.update_beliefs(b, t + 1, states[:, t + 1], conditions, responses[t])

                if N > 0:
                    with plate('responses_{}_{}'.format(b, t), N):
                        d = sample('d_{}_{}'.format(b, t),
                                   dist.Categorical(probs),
                                   infer={"enumerate": "parallel"})

                        sample('obs_{}_{}'.format(b, t),
                               dist.Bernoulli(logits=Vindex(logits)[..., d]),
                               obs=res)

    def guide(self):
        npar = self.agent.npar  # number of parameters
        nsub = self.runs  # number of subjects

        m_hyp = param('m_hyp', zeros(2*npar))
        st_hyp = param('scale_tril_hyp',
                       torch.eye(2*npar),
                       constraint=constraints.lower_cholesky)
        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :npar]
        unc_tau = hyp[..., npar:]

        trns_tau = biject_to(constraints.softplus_positive)

        c_tau = trns_tau(unc_tau)

        ld_tau = trns_tau.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        sample("mu_decentered", dist.Delta(unc_mu, event_dim=1))
        sample("var_tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('s_locs', torch.eye(npar).repeat(nsub, 1, 1),
                        constraint=constraints.lower_cholesky)

        alpha_12 = param('guide_alpha12', ones(nsub, 2, 2), constraint=constraints.softplus_positive)
        alpha_34 = param('guide_alpha34', ones(nsub, 2, 3), constraint=constraints.softplus_positive)

        with plate('subjects', nsub):
            sample("locs_decentered", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
            sample("probs_cond_12", dist.Dirichlet(alpha_12).to_event(1))
            sample("probs_cond_34", dist.Dirichlet(alpha_34).to_event(1))