#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate a sequence of context that would be used in all simulations.
Created on Mon Sep 23 13:33:50 2019

@author: Dimitrije Markovic
"""

import torch
from torch.distributions import Categorical

from numpy import arange, save

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', palette='colorblind', color_codes=True)

from setup_environment import *

# generate sequence of contexts
context = []
duration = []
offers = []
for k in range(blocks):
    if k == 0:
        cnt = Categorical(probs=pr_c).sample()
        off = Categorical(probs=pr_coo1[cnt, 0]).sample()
        drt = Categorical(probs=pr_cd[cnt]).sample()
    else:
        cnt = context[-1].clone()
        drt =  duration[-1].clone()

        if drt == 0:
            cnt = Categorical(probs=pr_dcc[0, cnt]).sample()
            off = Categorical(probs=pr_coo1[cnt, off]).sample()
            drt = Categorical(probs=pr_cd[cnt]).sample()
        else:
            drt -= 1

    context.append(cnt)
    duration.append(drt)
    
    offs = [off]
    for t in range(1, trials):
        offs.append(Categorical(probs=pr_coo1[cnt, offs[-1]]).sample())
    
    offs = torch.stack(offs)
    
    offers.append(offs.repeat(nsub, 1).transpose(dim0=1, dim1=0))
    
offers = torch.stack(offers)
context = torch.stack(context)

save('context_{}.npy'.format(blocks), context.numpy())
save('offers_{}.npy'.format(blocks), offers.numpy())

locs = (offers[:, 0, 0] == 0) + (offers[:, 0, 0] == 4) + (offers[:, 0, 0] == 2)
segs = arange(1, blocks + 1)
plt.plot(segs, context);
plt.plot(segs[locs], context[locs], 'bo');