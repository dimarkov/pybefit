import torch
import pyro.distributions as dist
from pyro import plate, sample, deterministic

def befit_likelihood(agent, data=None, task=None, blocks=1, trials=1, num_agents=1, **kwargs):
    # num_agents -> batch dimension - number of different subjects/agents
    # blocks -> number of experimental blocks
    # trials -> number of trials within each block

    data_absence = data is None
    task_absence = task is None

    if data_absence:
        assert not task_absence

    responses_all = []
    outcomes_all = []
    offers_all = []
    logits_all = []

    # define prior mean over model parametrs and subjects
    with plate('runs', num_agents):
        for b in range(blocks):
            responses_all.append([])
            outcomes_all.append([])
            logits_all.append([])
            offers_all.append([])
            for t in range(trials):
                # update single trial
                if data_absence: 
                    offers = task.get_offer(b, t)
                    if offers is not None:
                        offers_all[-1].append(offers)
                else:
                    offers = data['offers'][b, t] if data['offers'] is not None else None
                
                logits = agent.planning(b, t, offers=offers)
                logits_all[-1].append(logits)

                responses = None if data_absence else data['responses'][b, t]
                if data_absence:
                    mask = True
                else:
                    mask = True if data['mask'] is None else data['mask'][b, t].byte()
                
                responses = sample('obs_{}_{}'.format(b, t), dist.Categorical(logits=logits).mask(mask), obs=responses)
                responses_all[-1].append(responses)

                outcomes = task.update_environment(b, t, responses) if data_absence else data['outcomes'][b, t]
                outcomes_all[-1].append(outcomes)

                agent.update_beliefs(b, t, [responses, outcomes], mask=mask)
            
            responses_all[-1] = torch.stack(responses_all[-1])
            outcomes_all[-1] = torch.stack(outcomes_all[-1])
            logits_all[-1] = torch.stack(logits_all[-1])
            if len(offers_all[-1]) > 0:
                offers_all[-1] = torch.stack(offers_all[-1])

        deterministic('responses', torch.stack(responses_all))
        deterministic('outcomes', torch.stack(outcomes_all))
        deterministic('logits', torch.stack(logits_all))
        if len(offers_all[-1]) > 0:
            deterministic('offers', torch.stack(offers_all))