import pyro.distributions as dist
from pyro import plate, sample

def befit_likelihood(agent, data=None, num_blocks=1, num_trials=1, num_agents=1):
    # num_agents -> batch dimension - number of different subjects/agents
    # num_blocks -> number of experimental blocks
    # num_trials -> number of trials within each block

    # define prior mean over model parametrs and subjects
    with plate('runs', num_agents):
        for b in range(num_blocks):
            for t in range(num_trials):
                # update single trial
                offers = data['offers'][b, t]
                agent.planning(b, t, offers)

                logits = agent.logits[-1]

                outcomes = data['outcomes'][b, t]
                responses = data['responses'][b, t]
                mask = data['mask'][b, t]

                agent.update_beliefs(b, t, [responses, outcomes], mask=mask)

                sample('obs_{}_{}'.format(b, t), dist.Categorical(logits=logits).mask(mask.byte()), obs=responses)