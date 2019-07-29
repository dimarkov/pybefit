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
        self.stimulus = {'offers': [], 'outcomes': []}

    def simulate_experiment(self):
        """Simulates the experiment by iterating through all the blocks and trials,
           for each run in parallel. Here we generate responses and outcomes and 
           update agent's beliefs.
        """

        for b in range(self.nb):
            self.responses.append([None])
            self.stimulus['offers'].append([None])
            self.stimulus['outcomes'].append([None])
            for t in range(self.nt):
                #update single trial
                offers = self.env.get_offers(b, t)
                self.agent.planning(b, t, **offers)
                
                res = self.agent.sample_responses(b, t)
                response_outcome = self.env.update_environment(b, t, res)
                self.agent.update_beliefs(b, t, response_outcome)
                
                self.stimulus['offers'][-1].append(offers)
                self.stimulus['outcomes'][-1].append(response_outcome[-1])
                self.responses[-1].append(res)
                
#        self._format_stimuli_and_responses()
    
    def _format_stimuli_and_responses(self):
        
        for b in range(self.nb):
            self.stimulus['offers'][b] = torch.stack(self.stimulus['offers'][b][1:])
            self.stimulus['outcomes'][b] = torch.stack(self.stimulus['outcomes'][b][1:])
            self.responses[b] = torch.stack(self.responses[b][1:])
        
        self.responses = torch.stack(self.responses)
        self.stimulus['offers'] = torch.stack(self.stimulus['offers'])
        self.stimulus['outcomes'] = torch.stack(self.stimulus['outcomes'])
    
            



            

        

        
        
