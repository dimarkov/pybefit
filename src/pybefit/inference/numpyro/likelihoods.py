import numpyro.distributions as dist

from jax import lax
from numpyro import plate, sample, deterministic
from numpyro.contrib.control_flow import scan

def evolve_trials(agent, data):

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
        probs, outcomes, actions = evolve_trials(agent, xs)

        deterministic('outcomes', outcomes)

        with plate('num_agents', num_agents):
            with plate('num_trials', num_trials):
                sample('actions', dist.Categorical(logits=probs).to_event(1), obs=actions)
        
        # TODO: update agent parameters - learning

        return agent, None
    
    init_agent = agent
    scan(step_fn, init_agent, data, length=num_blocks)