#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains Bayesian agents for various experimental tasks
Created on Mon Jan 21 13:50:30 2019

@author: Dimitrije Markovic
"""

import warnings
import torch
from torch import zeros, ones
from opt_einsum import contract

from torch.distributions import Categorical

from ..agent import Discrete

softplus = torch.nn.functional.softplus

__all__ = [
        'HGFSocInf',
        'SGFSocInf',
        'ExplicitHMM',
        'ImplicitHMM',
        'ImplicitNBHMM'
]


class HGFSocInf(Discrete):

    def __init__(self, runs=1, blocks=1, trials=120):

        na = 2  # number of choices
        ns = 2  # number of states
        no = 2  # number of outcomes

        super(HGFSocInf, self).__init__(runs, blocks, trials, na, ns, no)

    def set_parameters(self, x=None, **kwargs):

        if x is not None:
            self.mu0_1 = torch.zeros_like(x[..., 0])
            self.mu0_2 = - softplus(x[..., 0] + 5.) - 2.
            self.mu0 = torch.stack([self.mu0_1, self.mu0_2], -1)
            self.pi0 = torch.ones_like(x[..., 2:4]) * 2.
            self.eta = (x[..., 1]-5).sigmoid()
            self.zeta = x[..., 2].sigmoid()
            self.beta = softplus(x[..., 3])
            self.bias = x[..., 4]
        else:
            self.mu0 = zeros(self.runs, 2)
            self.pi0 = ones(self.runs, 2)
            self.eta = .1*ones(self.runs)
            self.zeta = .95*ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.bias = zeros(self.runs)

        self.kappa = .5

        # set initial beliefs
        self.mu = [self.mu0]
        self.pi = [self.pi0]

        self.npar = 5
        self.offers = []
        self.logits = []

    def update_beliefs(self, b, t, response_outcomes, mask=None):

        if mask is None:
            mask = ones(self.runs)

        mu = []
        pi = []

        mu_pre = self.mu[-1]
        pi_pre = self.pi[-1]

        # 1st level
        # Prediction
        muhat = mu_pre[..., 0].sigmoid()

        #Update

        # 2nd level
        # Precision of prediction

        w1 = torch.exp(self.kappa*mu_pre[..., -1])

        # Updates
        pihat1 = pi_pre[..., 0]/(1 + pi_pre[..., 0] * w1)

        pi1 = pihat1 + mask * muhat * (1 - muhat)

        o = response_outcomes[-1][:, -2]  # observation
        wda = mask * ( (o + 1)/2 - muhat) / pi1
        mu.append(mu_pre[..., 0] + wda)
        pi.append(pi1)


        # Volatility prediction error
        da1 = mask * ((1 / pi1 + (wda)**2) * pihat1 - 1)

        # 3rd level
        # Precision of prediction
        pihat2 = pi_pre[..., 1] / (1. + pi_pre[..., 1] * self.eta)

        # Weighting factor
        w2 = w1 * pihat1 * mask

        # Updates
        pi2 = torch.clip(pihat2 + self.kappa**2 * w2 * (w2 + (2 * w2 - 1) * da1) / 2, min=1e-2)

        mu.append(mu_pre[..., 1] + self.kappa * w2 * da1 / (2 * pi2))
        pi.append(pi2)

        self.mu.append(torch.stack(mu, dim=-1))
        self.pi.append(torch.stack(pi, dim=-1))

    def planning(self, b, t, offers):

        sig = (self.kappa*self.mu[-1][..., 1]).exp() + self.pi[-1][..., 0]
        gamma = torch.sqrt(1 + 3.14159*sig/8)
        b_soc = (self.mu[-1][..., 0]/gamma).sigmoid()

        b_vis = offers

        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log()

        logits = self.beta * ln + self.bias
        logits = torch.stack([-logits, logits], -1)

        self.logits.append(logits)

    def sample_responses(self, b, t):

        cat = Categorical(logits=self.logits[-1])

        return cat.sample()

class SGFSocInf(Discrete):

    def __init__(self, runs=1, blocks=1, trials=120):

        na = 2  # number of choices
        ns = 2  # number of states
        no = 2  # number of outcomes

        super(SGFSocInf, self).__init__(runs, blocks, trials, na, ns, no)

    def set_parameters(self, x=None, **kwargs):

        if x is not None:
            self.rho1 = softplus(x[..., 0])
            self.h = (x[..., 1]-3).sigmoid()
            self.zeta = x[..., 2].sigmoid()
            self.beta = softplus(x[..., 3])
            self.bias = x[..., 4]
        else:
            self.rho1 = .1
            self.h = .05
            self.zeta = .5
            self.beta = 10

        self.sig0 = ones(self.runs)
        self.mu0 = zeros(self.runs)
        self.theta0 = zeros(self.runs)
        self.rho2 = 10.

        self.npar = 5
        # set initial beliefs
        self.mu = [self.mu0]
        self.sig = [self.sig0]

        self.theta = [self.theta0]
        self.logits = []
        self.offers = []

    def update_beliefs(self, b, t, response_outcomes, mask=None):

        if mask is None:
            mask = ones(self.runs)

        o = (response_outcomes[-1][:, -2] + 1.)/2.

    	# 1st level
    	# Prediction
        muhat = self.mu[-1].sigmoid()

    	# Update the precision and the expectation conditioned on the absence of a change
        sig_pre = self.sig[-1]
        sig1 = (sig_pre + self.rho1) / (1 + (sig_pre + self.rho1) * mask * muhat * (1-muhat))
        mu1 = self.mu[-1] + sig1 * mask * (o - muhat)


    	# Update the precision and the expectation conditioned on the presence of a change
        sig2 =  self.rho2
        mu2 = mask * sig2 * (o - .5)

    	# marginal observation likelihood
        gamma = torch.sqrt(1 + sig_pre * 3.14159 / 8)
        muhat2 = (self.mu[-1] / gamma).sigmoid()
        l1 = muhat2**o * (1 - muhat2)**(1 - o)
        l2 = .5

    	# Update posterior jump probability
        theta_pre = self.h * (1 - self.theta[-1])
        theta = l2 * theta_pre/( l1 * (1 - theta_pre) +  l2 * theta_pre)

        sig = (sig1 * sig2)/((1  - theta) * sig2 + theta * sig1)
        self.mu.append(sig*(mu1*(1-theta)/sig1 + mu2*theta/sig2))
        self.theta.append(theta)
        self.sig.append(sig)

    def planning(self, b, t, offers):
        gamma = torch.sqrt(1 + (self.sig[-1] + self.rho1) * 3.14159 / 8)
        theta = self.theta[-1]

        b_soc = (self.mu[-1]/gamma).sigmoid() * (1 - theta) + theta/2

        b_vis = offers

        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)

        ln = b_int.log() - (1 - b_int).log()

        logits = self.beta * ln + self.bias
        logits = torch.stack([-logits, logits], -1)

        self.logits.append(logits)

    def sample_responses(self, b, t):

        cat = Categorical(logits=self.logits[-1])

        return cat.sample()


class ExplicitHMM(Discrete):

    def __init__(self, runs=1, blocks=1, trials=1000, store=False):

        na = 3  # number of choices
        ns = 2  # number of states
        no = 4  # number of outcomes
        self.batch = (runs,)

        self.store = store  # switch for storing belief trajectories and components of expected free energy

        super(ExplicitHMM, self).__init__(runs, blocks, trials, na, ns, no)

        self.nd = 100

    def set_parameters(self, x=None, set_variables=True):
        self.npar = 8
        if x is not None:
            self.batch = x.shape[:-1]

        if x is not None:
            self.mu = 20. * x[..., 0].exp()
            self.phi = 1 + x[..., 1].exp()
            self.beta = 4. * x[..., 2].exp()
            self.lam = x[..., 3].sigmoid()
            self.s0 = x[..., 4].sigmoid()
            self.ph = (x[..., 5]+1.).sigmoid()*.5 + .5
            self.pl = (x[..., 6]-1.).sigmoid()*.5
            self.omega = x[..., 7].exp()
        else:
            self.mu = 20*ones(self.runs)
            self.phi = 25*ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.lam = ones(self.runs)
            self.ph = .8*ones(self.runs)
            self.pl = .2*ones(self.runs)
            self.s0 = .5*ones(self.runs)
            self.omega = ones(self.runs)

        if set_variables:
            cue = self.omega * (2 * self.lam - 1)
            self.U = torch.stack([-self.omega, self.omega, cue, cue], dim=-1)
            self.set_prior_beliefs()
            self.set_state_transition_matrix()
            self.set_observation_likelihood()

            self.offers = []
            self.obs_entropy = []
            self.pred_entropy = []
            self.values = []
            self.logits = []

    def set_prior_beliefs(self):
        ps = torch.stack([self.s0, 1 - self.s0], -1)

        k = torch.arange(float(self.nd))
        mu, phi = torch.broadcast_tensors(self.mu[..., None], self.phi[..., None])
        lrho = (mu + phi).log()
        lbinom = torch.lgamma(k + phi) - torch.lgamma(k + 1) - torch.lgamma(phi)
        self.pd = torch.softmax(lbinom + k * (mu.log() - lrho) + phi * (phi.log() - lrho), -1)

        P = self.pd.reshape(self.batch + (1, self.nd)) * ps.reshape(self.batch + (self.ns, 1))
        self.beliefs = [P]

    def set_state_transition_matrix(self):
        tm_dd = torch.diag(ones(self.nd-1), 1).expand(self.batch + (-1, -1)).clone()
        tm_dd[..., 0] = self.pd

        tm_ssd = zeros(self.ns, self.ns, self.nd)
        tm_ssd[..., 1:] = torch.eye(self.ns).reshape(self.ns, self.ns, 1)
        tm_ssd[..., 0] = (ones(self.ns, self.ns) - torch.eye(self.ns))/(self.ns - 1)

        self.tm_dd = tm_dd
        self.tm_ssd = tm_ssd

    def set_observation_likelihood(self):

        self.likelihood = zeros((self.batch) + (2, self.na, self.ns, self.no))
        ph = self.ph
        pl = self.pl

        self.likelihood[..., -1, 0, 2] = 1.
        self.likelihood[..., -1, 1, 3] = 1.

        self.likelihood[..., 0, 0, 0, 0] = 1-ph
        self.likelihood[..., 0, 0, 0, 1] = ph

        self.likelihood[..., 0, 1, 1, 0] = 1-ph
        self.likelihood[..., 0, 1, 1, 1] = ph

        self.likelihood[..., 0, 1, 0, 0] = 1-pl
        self.likelihood[..., 0, 1, 0, 1] = pl

        self.likelihood[..., 0, 0, 1, 0] = 1-pl
        self.likelihood[..., 0, 0, 1, 1] = pl

        self.likelihood[..., 1, 0, 1, 0] = 1-ph
        self.likelihood[..., 1, 0, 1, 1] = ph

        self.likelihood[..., 1, 1, 0, 0] = 1-ph
        self.likelihood[..., 1, 1, 0, 1] = ph

        self.likelihood[..., 1, 0, 0, 0] = 1-pl
        self.likelihood[..., 1, 0, 0, 1] = pl

        self.likelihood[..., 1, 1, 1, 0] = 1-pl
        self.likelihood[..., 1, 1, 1, 1] = pl

        self.entropy = - torch.sum(self.likelihood*(self.likelihood+1e-16).log(), -1)

    def update_beliefs(self, b, t, response_outcomes, mask=None):

        if mask is None:
            mask = ones(self.batch)

        res = response_outcomes[0]
        out = response_outcomes[1]

        offers = self.offers[-1]

        prior = self.beliefs[-1]
        tmp = self.likelihood[..., range(self.runs), offers, res, :, out]
        if tmp.dim() > 2:
            lik = 1. + mask[..., None] * (tmp.transpose(1, 0) - 1.)
        else:
            lik = 1. + mask[..., None] * (tmp - 1.)

        posterior = lik.reshape(self.batch + (self.ns, 1)) * prior
        norm = posterior.reshape(self.batch + (-1,)).sum(-1).reshape(self.batch + (1, 1))

        prediction = contract('...nij,klj,...nlj->...nki', self.tm_dd, self.tm_ssd, posterior/norm, backend='torch')

        if self.store:
            self.beliefs.append(prediction)
        else:
            self.beliefs = [prediction]

    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if self.store:
            self.offers.append(offers)
        else:
            self.offers = [offers]

        subs = list(range(self.runs))

        lik = self.likelihood[..., subs, offers, :, :, :]
        entropy = self.entropy[..., subs, offers, :, :]

        psd = self.beliefs[-1]

        marginal = contract('...naso,...nsd->...nao', lik, psd, backend='torch')

        H = - torch.sum(marginal * (marginal + 1e-16).log(), -1)
        V = contract('...nao,...no->...na', marginal, self.U, backend='torch')
        C = contract('...nas,...nsd->...na', entropy, psd, backend='torch')

        if self.store:
            self.values.append(V)
            self.obs_entropy.append(C)
            self.pred_entropy.append(H)

        self.logits.append(self.beta[..., None] * (V + H - C))

    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)

        return cat.sample()


class ImplicitHMM(Discrete):

    def __init__(self, runs=1, blocks=1, trials=1000, store=False):

        na = 3  # number of choices
        ns = 2  # number of states
        no = 4  # number of outcomes
        self.batch = (runs,)

        self.store = store  # switch for storing belief trajectories and components of expected free energy

        super(ImplicitHMM, self).__init__(runs, blocks, trials, na, ns, no)

    def set_parameters(self, x=None, set_variables=True):
        self.npar = 8
        if x is not None:
            self.batch = x.shape[:-1]

        if x is not None:
            self.p1 = (3 + x[..., 0]).sigmoid()
            self.p2 = (3 + x[..., 1]).sigmoid()
            self.g0 = (3 + x[..., 2]).sigmoid()
            self.beta = 4. * x[..., 3].exp()
            self.lam = x[..., 4].sigmoid()
            self.s0 = x[..., 5].sigmoid()
            self.ph = (x[..., 6]+1.).sigmoid()*.5 + .5
            self.pl = (x[..., 7]-1.).sigmoid()*.5
        else:
            self.p1 = .5 * ones(self.runs)
            self.p2 = .5 * ones(self.runs)
            self.g0 = .5 * ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.lam = ones(self.runs)
            self.ph = .8*ones(self.runs)
            self.pl = .2*ones(self.runs)
            self.s0 = .5*ones(self.runs)

        self.p = torch.stack([self.p1, self.p2, torch.ones_like(self.p1)], -1)
        self.g = torch.stack([self.g0, torch.ones_like(self.g0)], -1)

        if set_variables:
            self.U = torch.stack([-ones(self.batch), ones(self.batch), 2 * self.lam - 1, 2 * self.lam - 1], dim=-1)
            self.set_prior_beliefs()
            self.set_state_transition_matrix()
            self.set_observation_likelihood()

            self.offers = []
            self.obs_entropy = []
            self.pred_entropy = []
            self.values = []
            self.logits = []

    def set_prior_beliefs(self):
        ps = torch.stack([self.s0, 1 - self.s0], -1)

        pf = torch.tensor([1., 0., 0.])

        P = ps.reshape(self.batch + (self.ns, 1)) * pf
        self.beliefs = [P]

    def set_state_transition_matrix(self):
        tm_ff = torch.diag_embed(self.p) + \
                torch.diag_embed(self.g * (1 - self.p[..., :-1]), offset=1) + \
                torch.diag_embed((1 - self.g[..., :1]) * (1 - self.p[..., :1]), offset=2) + \
                torch.diag_embed(ones(1), offset=-2)

        tm_fss = zeros(self.p.shape[-1], self.ns, self.ns)
        tm_fss[:-1] = torch.eye(self.ns)
        tm_fss[-1] = (ones(self.ns, self.ns) - torch.eye(self.ns))/(self.ns - 1)

        self.tm_ff = tm_ff
        self.tm_fss = tm_fss

    def set_observation_likelihood(self):

        self.likelihood = zeros((self.batch) + (2, self.na, self.ns, self.no))
        ph = self.ph
        pl = self.pl

        self.likelihood[..., -1, 0, 2] = 1.
        self.likelihood[..., -1, 1, 3] = 1.

        self.likelihood[..., 0, 0, 0, 0] = 1-ph
        self.likelihood[..., 0, 0, 0, 1] = ph

        self.likelihood[..., 0, 1, 1, 0] = 1-ph
        self.likelihood[..., 0, 1, 1, 1] = ph

        self.likelihood[..., 0, 1, 0, 0] = 1-pl
        self.likelihood[..., 0, 1, 0, 1] = pl

        self.likelihood[..., 0, 0, 1, 0] = 1-pl
        self.likelihood[..., 0, 0, 1, 1] = pl

        self.likelihood[..., 1, 0, 1, 0] = 1-ph
        self.likelihood[..., 1, 0, 1, 1] = ph

        self.likelihood[..., 1, 1, 0, 0] = 1-ph
        self.likelihood[..., 1, 1, 0, 1] = ph

        self.likelihood[..., 1, 0, 0, 0] = 1-pl
        self.likelihood[..., 1, 0, 0, 1] = pl

        self.likelihood[..., 1, 1, 1, 0] = 1-pl
        self.likelihood[..., 1, 1, 1, 1] = pl

        self.entropy = - torch.sum(self.likelihood*(self.likelihood+1e-16).log(), -1)

    def update_beliefs(self, b, t, response_outcomes, mask=None):

        if mask is None:
            mask = ones(self.batch)

        res = response_outcomes[0]
        out = response_outcomes[1]

        offers = self.offers[-1]

        prior = self.beliefs[-1]
        tmp = self.likelihood[..., range(self.runs), offers, res, :, out]
        if tmp.dim() > 2:
            lik = 1. + mask[..., None] * (tmp.transpose(1, 0) - 1.)
        else:
            lik = 1. + mask[..., None] * (tmp - 1.)

        posterior = lik.reshape(self.batch + (self.ns, 1)) * prior
        norm = posterior.reshape(self.batch + (-1,)).sum(-1).reshape(self.batch + (1, 1))

        prediction = contract('...nfg,fsk,...nsf->...nkg', self.tm_ff, self.tm_fss, posterior/norm, backend='torch')

        if self.store:
            self.beliefs.append(prediction)
        else:
            self.beliefs = [prediction]

    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if self.store:
            self.offers.append(offers)
        else:
            self.offers = [offers]

        subs = list(range(self.runs))

        lik = self.likelihood[..., subs, offers, :, :, :]
        entropy = self.entropy[..., subs, offers, :, :]

        ps = self.beliefs[-1].sum(-1)

        marginal = contract('...naso,...ns->...nao', lik, ps, backend='torch')

        H = - torch.sum(marginal * (marginal + 1e-16).log(), -1)
        V = contract('...nao,...no->...na', marginal, self.U, backend='torch')
        C = contract('...nas,...ns->...na', entropy, ps, backend='torch')

        if self.store:
            self.values.append(V)
            self.obs_entropy.append(C)
            self.pred_entropy.append(H)

        self.logits.append(self.beta[..., None] * (V + H - C))

    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)

        return cat.sample()


class ImplicitNBHMM(ImplicitHMM):

    def __init__(self, order, runs=1, blocks=1, trials=1000, store=False):

        self.order = order
        super(ImplicitNBHMM, self).__init__(runs=runs, blocks=blocks, trials=trials, store=store)

    def set_parameters(self, x=None, set_variables=True):
        self.npar = 7
        if x is not None:
            self.batch = x.shape[:-1]

        if x is not None:
            self.mu = 20*x[..., 0].exp()
            self.beta = 4. * x[..., 1].exp()
            self.lam = x[..., 2].sigmoid()
            self.ph = (x[..., 3]+1.).sigmoid()*.5 + .5
            self.pl = (x[..., 4]-1.).sigmoid()*.5
            self.s0 = x[..., 5].sigmoid()
            self.omega = x[..., 6].exp()

        else:
            self.mu = 20 * ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.lam = ones(self.runs)
            self.ph = .8*ones(self.runs)
            self.pl = .2*ones(self.runs)
            self.s0 = .5*ones(self.runs)
            self.omega = ones(self.runs)

        self.p = self.mu / (self.mu + self.order)

        if set_variables:
            self.U = torch.stack([-ones(self.batch), ones(self.batch), 2 * self.lam - 1, 2 * self.lam - 1], dim=-1)
            self.set_prior_beliefs()
            self.set_state_transition_matrix()
            self.set_observation_likelihood()

            self.offers = []
            self.obs_entropy = []
            self.pred_entropy = []
            self.values = []
            self.logits = []

    def set_prior_beliefs(self):
        ps = torch.stack([self.s0, 1 - self.s0], -1)

        p = self.p.unsqueeze(-1)
        k = torch.arange(1., self.order + 1.)
        lbinom = torch.lgamma(self.order + ones(1)) - torch.lgamma(k) - torch.lgamma(self.order - k + 2)
        self.pi = lbinom.exp() * p ** (self.order - k + 1) * (1 - p) ** (k - 1)
        self.pi = torch.cat([self.pi, 1 - self.pi.sum(-1, keepdims=True)], -1)

        pf = self.pi

        P = ps.unsqueeze(-1) * pf.unsqueeze(-2)
        self.beliefs = [P]

    def set_state_transition_matrix(self):
        p = torch.stack([self.p]*self.order, -1)
        p = torch.cat([p, torch.zeros_like(self.p.unsqueeze(-1))], -1)
        tm_ff = torch.diag_embed(p) + torch.diag_embed(1 - p[..., :-1], offset=1)
        tm_ff[..., -1, :] = self.pi

        tm_fss = zeros(self.order + 1, self.ns, self.ns)
        tm_fss[:-1] = torch.eye(self.ns)
        tm_fss[-1] = (ones(self.ns, self.ns) - torch.eye(self.ns))/(self.ns - 1)

        self.tm_ff = tm_ff
        self.tm_fss = tm_fss

    def planning(self, b, t, offers):
        """Compute log probability of responses from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        if self.store:
            self.offers.append(offers)
        else:
            self.offers = [offers]

        subs = list(range(self.runs))

        lik = self.likelihood[..., subs, offers, :, :, :]
        entropy = self.entropy[..., subs, offers, :, :]

        ps = self.beliefs[-1].sum(-1)

        marginal = contract('...naso,...ns->...nao', lik, ps, backend='torch')

        H = - torch.sum(marginal * (marginal + 1e-16).log(), -1)
        U = self.omega.unsqueeze(-1) * self.U
        V = contract('...nao,...no->...na', marginal, U, backend='torch')
        C = contract('...nas,...ns->...na', entropy, ps, backend='torch')

        if self.store:
            self.values.append(V)
            self.obs_entropy.append(C)
            self.pred_entropy.append(H)

        self.logits.append(self.beta[..., None] * (V + H - C))
