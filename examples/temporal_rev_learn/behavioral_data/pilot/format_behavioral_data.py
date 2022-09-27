#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context = 'talk', style = 'white', color_codes = True)

home = os.path.expanduser('~')
cwd = os.getcwd()

fnames1 = [
	'OPVOL_10_053_180806_1729_WS.mat',
	'OPVOL_10_052_180806_1643_WS.mat',
	'OPVOL_10_051_180806_1558_WS.mat',
	'OPVOL_10_050_180806_1504_WS.mat',
	'OPVOL_10_039_180802_1429_WS.mat',
	'OPVOL_10_038_180802_1348_WS.mat',
	'OPVOL_10_037_180802_1304_WS.mat',
	'OPVOL_10_036_180802_1216_WS.mat',
	'OPVOL_10_035_180802_1125_WS.mat',
	'OPVOL_10_035_180802_1125_WS.mat',
	'OPVOL_10_034_180802_1002_WS.mat',
	'OPVOL_10_033_180801_1518_WS.mat',
	'OPVOL_10_033_180801_1518_WS.mat',
	'OPVOL_10_032_180801_1440_WS.mat'

]

fnames2 = [
	'OPVOL_10_049_180802_1822_WS.mat',
	'OPVOL_10_048_180802_1815_WS.mat',
	'OPVOL_10_047_180802_1645_WS.mat',
	'OPVOL_10_046_180802_1649_WS.mat',
	'OPVOL_10_045_180802_1550_WS.mat',
	'OPVOL_10_044_180802_1603_WS.mat',
	'OPVOL_10_043_180802_1510_WS.mat',
	'OPVOL_10_042_180802_1521_WS.mat',
	'OPVOL_10_041_180802_1437_WS.mat',
	'OPVOL_10_040_180802_1442_WS.mat'
]

fnames = fnames1 + fnames2
n_reg = len(fnames1)
n_irr = len(fnames2)
n = n_reg + n_irr

data_path = home + '/tudcloud/Shared/reversal/YoungAdults/data/pilot/'

from scipy import io
trials = 1000

offers = np.array([]).reshape(0, trials)
responses = np.array([]).reshape(0, trials)
outcomes = np.array([]).reshape(0, trials)
corrects = np.array([]).reshape(0, trials)

#    C = np.array([]).reshape(0, trials)
#    A = np.array([]).reshape(0, trials)
#    os.chdir(data_path)
#    n_subs = len(fnames[i])

os.chdir(data_path)
for j,f in enumerate(fnames):
    parts = f.split('_')
    tmp = io.loadmat(f)
    
    offers = np.vstack([offers, np.abs(tmp['random_lr']-2)])
    
    res = tmp['A'] - 1
    responses = np.vstack([responses, res])
    
    out = np.zeros(trials)
    hints = res[0] == 2
    nans = np.isnan(res[0])
    
    nothints = ~hints*~nans
    
    rew = tmp['R'][nothints][range(np.sum(nothints)), res[0, nothints].astype(np.int32)]
    out[nothints] = (rew + 1)/2
    out[nans] = np.nan
    out[hints] = tmp['S'][hints, 0] + 1
    outcomes = np.vstack([outcomes, out])
    
    correct = tmp['C']
    corrects = np.vstack([corrects, correct])
    
    
# #    RT = tmp['RT'][0]
os.chdir(cwd)
df_res = pd.DataFrame(data=responses)
df_res = df_res.rename_axis(index='subject', columns='trial')
df_res['condition'] = 'regular'
df_res.loc[n_reg:,'condition'] = 'irregular'


df_out = pd.DataFrame(data=outcomes)
df_out = df_out.rename_axis(index='subject', columns='trial')

df_corr = pd.DataFrame(data=corrects)
df_corr = df_corr.rename_axis(index='subject', columns='trial')

df_res.to_csv('responses.csv')
df_out.to_csv('outcomes.csv')
df_corr.to_csv('correct_responses.csv')
