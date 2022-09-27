#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
"""
Iterate through the tmp files in fit sims directory, collect all samples into one file.
@author: Dimitrije Markovic
"""
import glob
import numpy as np

# %%
samples = {}
for filename in glob.glob("fit_sims/tmp*.npz"):
    m = int(filename.split('.')[0].split('_')[-1][1:])
    tmp = np.load(filename, allow_pickle=True)['samples'].item()
    samples[m] = tmp
    print(tmp.keys())

# prior preferences 
P_o = [.1, .6, .15, .15] 

# save the file
np.savez('fit_sims/sims_mcomp_P-{}-{}-{}-{}.npz'.format(*P_o), samples=samples)
