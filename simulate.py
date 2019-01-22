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
    
    def __init__(self, environment, agent, runs = 1, blocks = 1, trials = 10):
        # set inital elements of the world to None        
        self.env = environment
        self.agent = agent
        
        self.runs = runs  # number of paralel runs of the experiment (e.g. number of subjects)
        self.nb = blocks  # number of experimental blocks
        self.nt = trials  # number of trials in each block
        
        # container for agents responses
        self.responses = [] 
        
        # container for choice outcomes or observations
        self.stimulus = []

    def simulate_experiment(self):
        """Simulates the experiment by iterating through all the blocks and trials, 
           for each run in parallel. Here we generate responses and outcomes and update 
           agent's beliefs.
        """

        for b in range(self.nb):
            self.responses.append([None])
            for t in range(self.nt):
                #update single trial
                res = self.responses[-1][-1]
                self.env.update_environment(b, t, res)
                
                self.stimulus.append(self.env.get_stimulus(b, t)) 
                                
                self.agent.update_beliefs(b, t, **self.stimulus[-1])
                self.agent.planning(b, t)
                
                self.responses[-1].append(self.agent.sample_responses(b, t))
        
        self._format_stimuli_and_responses()
    
    def _format_stimuli_and_responses(self):
        stimuli = {}
        for s in self.stimulus:
            names = s.keys()
            for name in names:
                stimuli.setdefault(name, [])
                stimuli[name].append(s[name])
        
        for name in stimuli.keys():
            stimuli[name] = torch.stack(stimuli[name]).reshape(self.nb, self.nt, self.runs)
        
        self.stimuli = stimuli
        
        for b in range(self.nb):
            self.responses[b] = torch.stack(self.responses[b][1:])
        
        self.responses = torch.stack(self.responses)
    
            



            

        

        
        
