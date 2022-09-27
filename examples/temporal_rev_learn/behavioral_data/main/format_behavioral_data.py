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

fnames11 = ['OPVOL_10_054_181205_1412_WS.mat',
           'OPVOL_10_055_181205_1451_WS.mat',
           'OPVOL_10_056_181205_1534_WS.mat',
           'OPVOL_10_057_181205_1622_WS.mat',
           'OPVOL_10_058_181205_1708_WS.mat',
           'OPVOL_10_059_181207_1408_WS.mat',
           'OPVOL_10_070_181213_1900_WS.mat',
           'OPVOL_10_071_181219_1401_WS.mat',
           'OPVOL_10_072_181219_1450_WS.mat',
           'OPVOL_10_073_181219_1510_WS.mat',
           'OPVOL_10_074_181219_1555_WS.mat',
           'OPVOL_10_075_181219_1610_WS.mat',
           'OPVOL_10_076_181219_1723_WS.mat',
           'OPVOL_10_077_181219_1716_WS.mat',
           'OPVOL_10_078_181219_1755_WS.mat'          
]

fnames21 = ['OPVOL_10_060_181207_1548_WS.mat',
           'OPVOL_10_061_181207_1626_WS.mat',
           'OPVOL_10_062_181213_1403_WS.mat',
           'OPVOL_10_063_181213_1502_WS.mat',
           'OPVOL_10_064_181213_1554_WS.mat',
           'OPVOL_10_065_181213_1647_WS.mat',
           'OPVOL_10_066_181213_1659_WS.mat',
           'OPVOL_10_067_181213_1747_WS.mat',
           'OPVOL_10_068_181213_1801_WS.mat',
           'OPVOL_10_069_181213_1852_WS.mat'
]


fnames12 = ['OPVOL_10_79_190620_1018_WS.mat',
           'OPVOL_10_81_190620_1217_WS.mat',
           'OPVOL_10_83_190620_1417_WS.mat',
           'OPVOL_10_85_190620_1557_WS.mat',
           'OPVOL_10_87_190620_1720_WS.mat',
           'OPVOL_10_89_190621_1508_WS.mat',
           'OPVOL_10_91_190621_1637_WS.mat',
           'OPVOL_10_93_190703_1038_WS.mat',
           'OPVOL_10_95_190703_1654_WS.mat',
           'OPVOL_10_99_190710_1523_WS.mat',
           'OPVOL_10_101_190710_1652_WS.mat',
           'OPVOL_10_103_190719_1353_WS.mat'
]

fnames22 = ['OPVOL_10_80_190620_1104_WS.mat',
           'OPVOL_10_82_190620_1241_WS.mat',
           'OPVOL_10_84_190620_1501_WS.mat',
           'OPVOL_10_86_190620_1634_WS.mat',
           'OPVOL_10_88_190621_1421_WS.mat',
           'OPVOL_10_90_190621_1551_WS.mat',
           'OPVOL_10_92_190621_1754_WS.mat',
           'OPVOL_10_94_190703_1439_WS.mat',
           'OPVOL_10_96_190703_1738_WS.mat',
           'OPVOL_10_98_190705_1632_WS.mat',
           'OPVOL_10_100_190710_1606_WS.mat',
           'OPVOL_10_102_190710_1733_WS.mat',
           'OPVOL_10_104_190719_1557_WS.mat'
]

fnames = fnames11 + fnames12 + fnames21 + fnames22
n_reg = len(fnames11) + len(fnames12)
n_irr = len(fnames21) + len(fnames22)
n = n_reg + n_irr

data_path = home + '/tudcloud/Shared/reversal/YoungAdults/data/'

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
    
    rew = tmp['R'][nothints][range(np.sum(nothints)), res[0, nothints].astype(np.long)]
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
