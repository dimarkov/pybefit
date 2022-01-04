#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contain reinforcement learning agents for various experimental tasks
Created on Mon Jan 21 13:50:01 2019

@author: Dimitrije Markovic
"""

import torch
from torch import ones, zeros, zeros_like, arange
from torch.distributions import Categorical

from .agent import Discrete

__all__ = [
        'RLSocInf',
        'RLTempRevLearn'
]

softplus = torch.nn.Softplus()

class RLSocInf(Discrete):
    """Reiforcement learning agent (using Rescorla-Wagner learning rule) for 
    the Social Influence task.
    """

    def __init__(self, runs=1, blocks=1, trials=1):

        na = 2  # number of actions
        ns = 2  # number of states
        no = 2  # number of outcomes
        super(RLSocInf, self).__init__(runs, blocks, trials, na, ns, no)

    def set_parameters(self, x=None, **kwargs):

        if x is not None:
            self.alpha = x[..., 0].sigmoid()
            self.zeta = x[..., 1].sigmoid()
            self.beta = softplus(x[..., 2])
            self.bias = x[..., 3]
        else:
            self.alpha = .25*ones(self.runs)
            self.zeta = .95*ones(self.runs)
            self.beta = 10.*ones(self.runs)

        self.V0 = zeros(self.runs)
        self.npar = 4

        # set initial value vector
        self.values = [self.V0]
        self.offers = []
        self.logits = []

    def update_beliefs(self, b, t, response_outcomes, mask=None):

        if mask is None:
            mask = ones(self.runs)

        V = self.values[-1]
        o = response_outcomes[-1][:, -2]

        # update choice values
        self.values.append(V + mask * self.alpha * (o - V))

    def planning(self, b, t, offers):
        """Compute response probability from values."""
        V = self.values[-1]
        b_soc = (1 + V)/2
        b_vis = offers
        b_int = b_soc * self.zeta + b_vis * (1 - self.zeta)
        ln = b_int.log() - (1 - b_int).log()

        logits = self.beta * ln + self.bias
        logits = torch.stack([-logits, logits], -1)
        self.logits.append(logits)

    def sample_responses(self, b, t):
        cat = Categorical(logits=self.logits[-1])

        return cat.sample()


class RLTempRevLearn(Discrete):
    """here we implement a reinforcement learning agent for the temporal
    reversal learning task.
    """

    def __init__(self, runs=1, blocks=1, trials=1):

        na = 3  # number of actions
        ns = 2  # number of states
        no = 4  # number of outcomes
        super(RLTempRevLearn, self).__init__(runs, blocks, trials, na, ns, no)

    def set_parameters(self, x=None, set_variables=True):

        if x is not None:
            self.alpha = x[..., 0].sigmoid()
            self.kappa = x[..., 1].sigmoid()
            self.beta = x[..., 2].exp()
            self.theta = x[..., 3]
            self.batch = x.shape[:-1]
        else:
            self.alpha = .25*ones(self.runs)
            self.kappa = ones(self.runs)
            self.beta = 10.*ones(self.runs)
            self.theta = zeros(self.runs)
            self.batch = (self.runs,)

        if set_variables:
            self.V0 = zeros(self.batch + (self.na,))
            self.V0[..., -1] = self.theta
            self.npar = 4

            # set initial value vector
            self.values = [self.V0]
            self.offers = []
            self.outcomes = []
            self.logits = []

    def update_beliefs(self, b, t, response_outcome, mask=None):

        if mask is None:
            mask = ones(self.runs)

        V = self.values[-1]

        res = response_outcome[0]
        obs = response_outcome[1]

        hints = res == 2  # exploratory choices
        nothints = ~hints
        lista = arange(self.runs)

        alpha = self.alpha[..., nothints]
        kappa = self.kappa[..., nothints]

        # update choice values
        V_new = zeros_like(V)
        V_new[..., -1] = self.theta

        if torch.get_default_dtype() == torch.float32:
            rew = 2.*obs[nothints].float() - 1.
        else:
            rew = 2.*obs[nothints].double() - 1.

        choices = res[nothints]
        loc = lista[nothints]
        V1 = V[..., loc, choices]
        V_new[..., loc, choices] = V1 + alpha * mask[nothints] * (rew - V1)

        V2 = V[..., loc, 1 - choices]
        V_new[..., loc, 1 - choices] = V2 - alpha * kappa * mask[nothints] * (rew + V2)

        cue = obs[hints] - 2
        loc = lista[hints]
        V_new[..., loc, cue] = 1.
        V_new[..., loc, 1 - cue] = - self.kappa[..., hints]
        self.values.append(V_new)

    def planning(self, b, t, offers):
        """Compute response probability from stimuli values for the given offers.
           Here offers encode location of stimuli A and B.
        """
        self.offers.append(offers)
        loc1 = offers == 0
        loc2 = ~loc1
        V1 = self.values[-1]
        if loc2.any():
            V2 = torch.stack([V1[..., 1], V1[..., 0], V1[..., -1]])
            V = torch.where(loc1[:, None], V1, V2)
        else:
            V = V1

        self.logits.append(self.beta[..., None] * V)

    def sample_responses(self, b, t):
        logits = self.logits[-1]
        cat = Categorical(logits=logits)

        return cat.sample()
