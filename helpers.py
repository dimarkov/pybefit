#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of helper functions 

@author: Dimtirije Markovic
"""
import numpy as np
import torch

def offer_state_mapping(L, no, nd=2):
    """
    Compute the state transition matrix for a given offer.
    L -> size of the state space along one dimension
    no -> number of offers
    nd -> number of dimensions
    """
    ns = L**nd
    states = np.arange(ns).astype(int)
    state_trans = np.zeros((no, ns, ns))
    #define option impact on the state space
    #A -> (1,0), B -> (0,1), Ab -> (1,-1), Ba -> (-1,1)
    opt = np.array([[1, 0], [0, 1], [1, -1], [-1, 1]])
    state_indices = np.indices((L, )*nd).T
    for i, o in enumerate(opt):
        new_states = state_indices+o
        new_states[new_states > L-1] = L-1
        new_states[new_states < 0] = 0
        new_states = np.ravel_multi_index(new_states.T, (L, )*nd).flatten()
        state_trans[i, states, new_states] = 1
    
    return torch.from_numpy(state_trans)