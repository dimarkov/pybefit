"""This module contains the Simulator class that defines interactions between
the environment and the agent. It also keeps track of all generated observations
and responses. To initiate it one needs to provide the environment
class and the agent class that will be used for the experiment.
"""
from torch import zeros, long

class Simulator(object):

    def __init__(self, environment, agent, runs=1, mini_blocks=1, trials=10):
        self.env = environment
        self.agent = agent

        self.runs = runs  # number of paralel runs of the experiment (e.g. number of subjects)
        self.nb = mini_blocks  # number of mini-blocks
        self.nt = trials  # number of trials in each mini-block

        # container for agents responses
        self.responses = zeros(self.runs, self.nb, self.nt)-1
        self.outcomes = zeros(self.runs, self.nb, self.nt, dtype=long)-1

    def simulate_experiment(self):
        """Simulates the experiment by iterating through all the mini-blocks and trials,
           for each run in parallel. Here we generate responses and outcomes and update
           agent's beliefs.
        """

        for b in range(self.nb):
            conditions = self.env.conditions[..., b]
            for t in range(self.nt + 1):
                # update single trial
                states = self.env.states[:, b, t]

                if t == 0:
                    self.agent.update_beliefs(b, t, states, conditions)
                else:
                    self.agent.update_beliefs(b, t, states, conditions, self.responses[:, b, t-1])

                if t < self.nt:
                    self.agent.plan_actions(b, t)

                    res = self.agent.sample_responses(b, t)

                    self.env.update_environment(b, t, res)

                    self.responses[:, b, t] = res
                    self.outcomes[:, b, t] = self.env.sample_outcomes(b, t)












