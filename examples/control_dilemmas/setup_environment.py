#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of relevant variables for setting task environment
Created on Mon Sep 23 13:33:50 2019

@author: Dimitrije Markovic
"""

import torch
from torch import ones, zeros, eye

blocks = 200  # number of experimental blocks
trials = 5  # number of trials in each block
nsub = 100  # number of simulated subjects

nc = 3  # number of contexts
no = 6  # number of offers/stimuli configurations
nd = 20  # max duration
ns = 6  # number of arm types
na = 4  # number of arms/choices
nf = 3  # number of features (null, red, blue)

# Define environment
pr_c = torch.tensor([1., 1., 1.])/3.

# banidt types in different contexts
pr_coo1 = zeros(nc, no, no)
pr_coo1[0, [0, 2, 4], 1] = 1.
pr_coo1[0, [1, 3, 5], 0] = 1.
pr_coo1[1, [0, 2, 4], 3] = 1.
pr_coo1[1, [1, 3, 5], 2] = 1.
pr_coo1[2, [0, 2, 4], 5] = 1.
pr_coo1[2, [1, 3, 5], 4] = 1.

pr_co = pr_coo1.sum(-2)
pr_co /= pr_co.sum(-1, keepdim=True)

# outcome likelihood
rho = .8
pr_pr = (1-rho)*ones(ns, nf)/(nf - 1)
pr_pr[0, 0] = rho
pr_pr[1, 1] = rho
pr_pr[2, 2] = rho
pr_pr[3, :] = 0.; pr_pr[3, 0] = 1.
pr_pr[4, :] = 0.; pr_pr[4, 1] = 1.
pr_pr[5, :] = 0.; pr_pr[5, 2] = 1.

pr_pr /= pr_pr.sum(-1, keepdim=True)

priors = {'offers': pr_co,
          'locations': ones(na)/na,
          'probs': pr_pr}

# context-duration transition probability
pr_dcc = zeros(nd, nc, nc)
pr_dcc[0] = torch.ones(nc, nc)/(nc - 1)
pr_dcc[0, range(nc), range(nc)] = 0.
pr_dcc[1:] = eye(nc).repeat(nd-1, 1, 1)

pr_cd = zeros(nc, nd)
pr_cd[:, 4] = 1.


# location-outcome likelihood
pr_all = eye(na).reshape(na, 1, na).repeat(1, na, 1)
transitions = {'locations': pr_all}