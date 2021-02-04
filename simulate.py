"""This module contains the Simulator class that defines interactions between
the environment and the agent. It also keeps track of all generated observations
and responses. To initiate it one needs to provide the environment
class and the agent class that will be used for the experiment.
"""
import torch
zeros = torch.zeros

__all__ = [
        'Simulator'
]


class Simulator(object):

    def __init__(self, environments, agents, blocks=1, trials=10):
        # set inital elements of the world to None
        self.envs = environments
        self.agents = agents

        assert len(self.envs) == len(self.agents)

        self.nb = blocks  # number of experimental blocks
        self.nt = trials  # number of trials in each block

        self.stimuli = {}
        self.responses = {}
        for i in range(len(self.envs)):
            key = 'pair_{}'.format(i)
            # container for choice outcomes or observations
            self.stimuli[key] = {'offers': [], 'outcomes': []}
            # container for agents responses
            self.responses[key] = []

    def simulate_experiment(self):
        """Simulates the experiment by iterating through all the blocks and trials,
           for each run in parallel. Here we generate responses and outcomes and
           update agent's beliefs.
        """

        for b in range(self.nb):
            for key in self.stimuli.keys():
                self.responses[key].append([None])
                self.stimuli[key]['offers'].append([None])
                self.stimuli[key]['outcomes'].append([None])
            for t in range(self.nt):
                # update single trial
                for agent, env, key in zip(self.agents, self.envs, self.stimuli.keys()):
                    offers = env.get_offers(b, t)
                    agent.planning(b, t, offers)

                    res = agent.sample_responses(b, t)
                    response_outcome = env.update_environment(b, t, res)
                    agent.update_beliefs(b, t, response_outcome)

                    self.stimuli[key]['offers'][-1].append(offers)
                    self.stimuli[key]['outcomes'][-1].append(response_outcome[-1])
                    self.responses[key][-1].append(res)

    def format_stimuli_and_responses(self):

        for key in self.stimuli:
            for b in range(self.nb):
                self.stimuli[key]['offers'][b] = torch.stack(self.stimuli[key]['offers'][b][1:])
                self.stimuli[key]['outcomes'][b] = torch.stack(self.stimuli[key]['outcomes'][b][1:])
                self.responses[key][b] = torch.stack(self.responses[key][b][1:])

            self.responses[key] = torch.stack(self.responses[key])
            self.stimuli[key]['offers'] = torch.stack(self.stimuli[key]['offers'])
            self.stimuli[key]['outcomes'] = torch.stack(self.stimuli[key]['outcomes'])

        return self.stimuli, self.responses
