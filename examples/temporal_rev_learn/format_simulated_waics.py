#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
"""
Iterate through the files containing simulated waic results and combine them 
into one dictionary and save to a single file.
@author: Dimitrije Markovic
"""

import glob
import numpy as np
from collections import defaultdict

# %%
res = {'max': {}, 'min': {}}
for filename in glob.glob("waic_sim_m*.npz"):
    s = filename.split('.')[0].split("_")[2:]
    tmp = np.load(filename, allow_pickle=True)['waic'].item()

    for _nu_inf in tmp:
        for _nu_true in tmp[_nu_inf]:
            if s[0] == 'max':
                nu_true = _nu_true
                if int(s[1]) == 0:
                    nu_inf = _nu_inf
                else:
                    nu_inf = 10 + _nu_inf - 1
            else:
                nu_true = 10 + _nu_true - 1
                if int(s[1]) == 0:
                    nu_inf = 10 + _nu_inf - 1
                else:
                    nu_inf = _nu_inf
            if  nu_inf in res[s[0]]:
                res[s[0]][nu_inf][nu_true] = tmp[_nu_inf][_nu_true]
            else:
                res[s[0]][nu_inf] = {nu_true: tmp[_nu_inf][_nu_true]}
            print(tmp[_nu_inf][_nu_true].shape)

# if main fail already exists merge results to create larger sample size

# save waic scores U=jnp.array([0, 1., 0., 0.])
np.savez('waic_sim_all2.npz', waic=res)
# %%
