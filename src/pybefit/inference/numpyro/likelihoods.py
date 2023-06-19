import numpyro.distributions as dist
import jax.numpy as jnp

from jax import lax, random
from numpyro import plate, sample, deterministic
from numpyro.contrib.control_flow import scan

def pymdp_evolve_trials(agent, data):

    def step_fn(carry, xs):
        empirical_prior = carry
        outcomes = xs['outcomes']
        qs = agent.infer_states(outcomes, empirical_prior)
        q_pi, _ = agent.infer_policies(qs)

        probs = agent.action_probabilities(q_pi)

        actions = xs['actions']
        empirical_prior = agent.update_empirical_prior(actions, qs)
        
        #TODO: if outcomes and actions are None, generate samples
        return empirical_prior, (probs, outcomes, actions)

    prior = agent.D
    _, res = lax.scan(step_fn, prior, data)

    return res

def pymdp_likelihood(agent, data=None, num_blocks=1, num_trials=1, num_agents=1):
    # Na -> batch dimension - number of different subjects/agents
    # Nb -> number of experimental blocks
    # Nt -> number of trials within each block

    def step_fn(carry, xs):
        agent = carry
        probs, outcomes, actions = pymdp_evolve_trials(agent, xs)

        deterministic('outcomes', outcomes)

        with plate('num_agents', num_agents):
            with plate('num_trials', num_trials):
                sample('actions', dist.Categorical(probs=probs).to_event(1), obs=actions)
        
        # TODO: update agent parameters - learning

        return agent, None
    
    init_agent = agent
    scan(step_fn, init_agent, data, length=num_blocks)


def befit_evolve_trials(key, agent, init_beliefs, b, trials, data=None, task=None):
    data_absence = data is None
    if task is None:
        assert not data_absence

    def step_fn(carry, t):
        _beliefs, key = carry
        key, _key = random.split(key)
        if data_absence:
            offers = task.get_offer(b, t)
            mask = True
        else:
            offers = None if data['offers'] is None else data['offers'][t]
            mask = True if data['mask'] is None else data['mask'][t]

        logits = agent.planning(b, t, offers=offers, beliefs=_beliefs)

        responses = agent.sample_responses(b, t, logits, key=_key) if data_absence else data['responses'][t]
        outcomes = task.update_environment(b, t, responses) if data_absence else data['outcomes'][t]
        beliefs = agent.update_beliefs(b, t, [responses, outcomes], beliefs=_beliefs, mask=mask)

        return (beliefs, key), (logits, beliefs, responses, outcomes, offers)
    
    (last_beliefs, _), res = lax.scan(step_fn, (init_beliefs, key), jnp.arange(trials))

    return last_beliefs, *res


def befit_likelihood(agent, data=None, task=None, blocks=1, trials=1, num_agents=1, seed=0, **kwargs):
    # num_agents -> batch dimension - number of different subjects/agents
    # blocks -> number of experimental blocks
    # trials -> number of trials within each block

    data_absence = data is None
    if data_absence:
        assert task is not None

    keys = random.split(random.PRNGKey(seed), blocks)
    # define prior mean over model parametrs and subjects
    def step_fn(carry, xs):
        init_beliefs = carry
        b, key, block_data = xs
        if not data_absence:
            mask = True if block_data['mask'] is None else data['mask']
        else:
            mask = True
        
        last_beliefs, logits, beliefs, responses, outcomes, offers = befit_evolve_trials(
            key,
            agent, 
            init_beliefs, 
            b, 
            trials, 
            data=block_data, 
            task=task
        )

        with plate('runs', num_agents):
            with plate('num_trials', trials):
                sample('responses', dist.Categorical(logits=logits).mask(mask), obs=responses)

        deterministic('outcomes', outcomes)
        deterministic('logits', logits)
        deterministic('beliefs', beliefs)
        if offers is not None:
            deterministic('offers', offers)

        return last_beliefs, None

    scan(step_fn, agent.get_beliefs, (jnp.arange(blocks), keys, data), length=blocks)