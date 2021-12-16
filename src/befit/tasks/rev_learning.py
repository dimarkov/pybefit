#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains various experimental environments used for testing
human behavior.

Created on Thu Feb  22 14:50:01 2018

@author: Dimitrije Markovic
"""

import torch
from torch.distributions import Categorical

__all__ = [
        'SocialInfluence',
        'TempRevLearn'
]

class SocialInfluence(object):
    """Implementation of the social learning task
    """

    def __init__(self, stimuli, nsub=1, blocks=1, trials=120):

        self.trials = trials
        self.nsub = nsub

        # set stimuli
        self.stimuli = stimuli


    def get_offers(self, block, trial):
        offers = self.stimuli['offers'][block, trial]

        return offers

    def update_environment(self, block, trial, responses):
        """Generate stimuli for the current block and trial
        """
        reliability = self.stimuli['reliability'][block, trial]
        reward_if_follow = 2*reliability - 1
        reward_if_reject = - reward_if_follow
        reward = torch.stack([reward_if_reject, reward_if_follow], -1)

        # list of different outcomes
        # 0 - is advice reliable (0,1)
        # 1 - reward if one would follow the advice
        # 2 - reward dependent on agents choice

        outcomes = torch.stack([reward_if_follow, reward[range(len(responses)), responses]], -1)

        return [responses, outcomes]


class TempRevLearn(object):
    """Implementation of the temporal reversal learning task.
    """

    def __init__(self, stimuli=None, nsub=1, blocks=1, trials=1000):
        self.trials = trials
        self.nsub = nsub

        # set stimuli
        self.stimuli = stimuli

    def likelihood(self, block, trial, responses):
        raise NotImplementedError

    def update_states(self, block, trial):
        raise NotImplementedError

    def get_offers(self, block, trial):

        if self.stimuli is not None:
            offers = self.stimuli['offers'][block, trial]
        else:
            cat = Categorical(logits=torch.zeros(2))
            offers = cat.sample((self.nsub,))

        return offers

    def update_environment(self, block, trial, responses):

        if self.stimuli is not None:
            outcomes = self.stimuli['rewards'][block, trial, range(self.nsub), responses]
            response_outcome = [responses, outcomes]
        else:
            # update states
            self.update_states(block, trial)
            # generate outcomes
            probs = self.likelihood(block, trial, responses)
            cat1 = Categorical(probs=probs)
            response_outcome = [responses, cat1.sample((self.nsub,))]

        return response_outcome
