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

# %%
res = {}
for filename in glob.glob("waic_sim_m*.npz"):
    tmp = np.load(filename, allow_pickle=True)['waic'].item()
    for m_inf in tmp:
        res[m_inf] = {}
        for m_true in tmp[m_inf]:
            res[m_inf][m_true] = tmp[m_inf][m_true]

# save waic scores for U=[0, 1., 0., 0.]
np.savez('fit_waic_sims/waic_sim_all_U-0-1-0-0_g4.npz', waic=res)

# save waic scores for U=jnp.array([1.5, 2., 0., 0.])
# np.savez('waic_sim_all_U-1.5-2-0-0.npz', waic=res)
# %%
