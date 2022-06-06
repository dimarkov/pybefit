#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various auxiliary functions for data formating and statistics
@author: Dimitrije Markovic
"""

import numpy as np
import pandas as pd

def load_data():
    df_res = pd.read_csv('main/responses.csv').set_index('subject')
    df_res['experiment'] = 'main'
    _df_res = pd.read_csv('pilot/responses.csv').set_index('subject')
    _df_res['experiment'] = 'pilot'
    df_res = pd.concat([df_res, _df_res], ignore_index=True)
    responses = df_res.values[:, :-2].T.astype(float)

    df_out = pd.read_csv('main/outcomes.csv').set_index('subject')
    df_out['experiment'] = 'main'
    df_out['condition'] = df_res['condition']

    _df_out = pd.read_csv('pilot/outcomes.csv').set_index('subject')
    _df_out['experiment'] = 'pilot'
    _df_out['condition'] = _df_res['condition']
    df_out = pd.concat([df_out, _df_out], ignore_index=True)
    outcomes = df_out.values[:, :-2].T.astype(float)

    df_cor = pd.read_csv('main/correct_responses.csv').set_index('subject')
    df_cor['experiment'] = 'main'
    df_cor['condition'] = df_res['condition']

    _df_cor = pd.read_csv('pilot/correct_responses.csv').set_index('subject')
    _df_cor['experiment'] = 'pilot'
    _df_cor['condition'] = df_res['condition']
    df_cor = pd.concat([df_cor, _df_cor], ignore_index=True)
    corrects = df_cor.values[:, :-2].T.astype(float)

    ns_reg = (df_res['condition'] == 'regular').sum()   # number of subjects in the regular reversals group
    ns_irr = (df_res['condition'] == 'irregular').sum()  # number of subjects in the irregular reversals group

    nsub = responses.shape[-1]

    assert np.all((np.isnan(responses) + (responses == 2)) == np.isnan(corrects))

    corrects_all = np.where(np.isnan(responses), 0., corrects)

    mask_all = ~np.isnan(responses)
    responses_all = np.nan_to_num(responses)
    outcomes_all = np.nan_to_num(outcomes)
    
    return outcomes_all, responses_all, mask_all, (nsub, ns_reg, ns_irr), corrects_all, df_res.loc[:,'condition':'experiment']

def trials_until_correct(correct, state, τ=2):
    # mark trial on which state switch occured
    state_switch = np.insert(np.abs(np.diff(state, axis=-1)), 0, 0, axis=-1)

    count = np.zeros(state_switch.shape[0])  # counter trials since the last switch
    allowed = np.ones(state_switch.shape[0])
    cum_corr = np.zeros(correct.shape[:-1])  # counter for corect responses in a sequence
    tuc = np.zeros(correct.shape)  # trials until correct
    for t in range(state_switch.shape[-1]):
        # increase counter if state did not switch, otherwise reset to 1
        count = (count + 1) * (1 - state_switch[..., t]) + state_switch[..., t]

        # if choice was correct increase by 1, otherwise reset to 0
        cum_corr = (cum_corr + 1) * correct[..., t]

        # check if the number of correct choices matches the threshold value
        at_threshold = (cum_corr == τ) * allowed
        allowed = (1 - at_threshold) * allowed * (1 - state_switch[..., t]) + state_switch[..., t]

        # update only valid dimensions for which count is larger than the threshold
        valid = (count >= τ)
        # mark count for valid dimensions (participants) which satisfy the condition
        # all the other elements are set to NaN
        tuc[..., t] = np.where(valid * at_threshold, count, np.nan)

    cum_tuc = np.nancumsum(tuc, axis=-1)
    for i, s in enumerate(state_switch):
        tau = np.diff(np.insert(np.nonzero(s), 0, 0))  # between reversal duration
        d_tuc = np.diff(np.insert(cum_tuc[i, s.astype(bool)], 0, 0))  # between reversal tuc
        # if change in tuc is zero add tau as maximal tuc
        loc = d_tuc == 0  # where tuc did not change
        trials_before_switch = np.arange(tuc.shape[-1])[s.astype(bool)] - 1
        tuc[i, trials_before_switch[loc]] = tau[loc]

    return tuc

def trials_until_explore(explore, state):
    # mark trial on which state switch occured
    state_switch = np.insert(np.abs(np.diff(state, axis=-1)), 0, 0, axis=-1)

    count = np.zeros(state_switch.shape[0])  # counter trials since the last switch
    cum_explore = np.zeros(explore.shape[:-1])
    trials_until_explore = np.zeros(explore.shape)
    for t in range(state_switch.shape[-1]):
        # increase counter if state did not switch, otherwise reset to 1
        count = (count + 1) * (1 - state_switch[..., t]) + state_switch[..., t]

        # if choice was exploratory increase by 1
        cum_explore = cum_explore + explore[..., t]

        # reset count to zero after the state switch
        cum_explore = cum_explore * (1 - state_switch[..., t])

        # check if the choice was to explore
        at_explore = cum_explore == 1

        # mark count for dimensions (participants) which satisfy the condition
        # all the other elements are set to NaN
        trials_until_explore[..., t] = np.where(at_explore, count, np.nan)

    # retrun mean trials until correct across the experiment for each participant and sample
    return trials_until_explore

def running_mean(x, ws=20):
    z = np.insert(x, 0, np.nan, axis=-1)
    cumsum = np.nancumsum(z, -1)
    count = np.cumsum(~np.isnan(z), -1)
    return (cumsum[..., ws:] - cumsum[..., :-ws]) / (count[..., ws:] - count[..., :-ws])

def odds(x):
    return x/(1 - x)

def performance(correct, ws=100):
    return odds(running_mean(correct.T, ws=ws))
