import numpyro.distributions as dist
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr

from jax import lax, random
from numpyro import plate, sample, deterministic
from numpyro.contrib.control_flow import scan
from numpyro import prng_key

from jaxtyping import Array

def multiaction_to_category(unique_multiactions: Array, multiaction: Array):
    return jnp.argmax(jnp.all(unique_multiactions == jnp.expand_dims(multiaction, -2), -1), -1)

def local_scan(f, init, xs, length=None, axis=0):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    if y is not None:
       ys.append(y)
  
  ys = None if len(ys) < 1 else jtu.tree_map(lambda *x: jnp.stack(x,axis=axis), *ys)

  return carry, ys

def pymdp_evolve_trials(agent, data, task, num_trials):

    data_absence = data is None
    if data_absence:
        assert task is not None # if there is no experimental data task env has to be passed

    def step_fn(carry, t):
        actions = carry['multiactions']
        outcomes = carry['outcomes']
        task = carry['task']
        beliefs = agent.infer_states(
            outcomes,
            carry['args'][0],
            past_actions=actions,
            qs_hist=carry['args'][1],
            mask=None # TODO: add masked observations (e.g. for no response-outcome trials)
        )
        q_pi, G = agent.infer_policies(beliefs) # what to do with G?
        if data_absence:
            keys = jr.split(prng_key(), agent.batch_size + 1)
            actions_t = agent.sample_action(q_pi, rng_key=keys[:-1])
            keys =  jr.split(keys[-1], agent.batch_size)
            outcome_t, task = task.step(keys, actions=actions_t)
        else:
            actions_t = data['multiactions'][..., t, :]
            outcome_t = jtu.tree_map(lambda x: x[..., t + 1], data['outcomes'])

        action_probs_t = agent.multiaction_probabilities(q_pi)
        outcomes = jtu.tree_map(
           lambda prev_o, new_o: jnp.concatenate(
               [prev_o, jnp.expand_dims(new_o, -1)], 
               -1), outcomes, outcome_t
          )

        if actions is not None:
          actions = jnp.concatenate([actions, jnp.expand_dims(actions_t, -2)], -2)
        else:
          actions = jnp.expand_dims(actions_t, -2)

        args = agent.update_empirical_prior(actions_t, beliefs)

        # args = (pred_{t+1}, [post_1, post_{2}, ..., post_{t}])
        # beliefs =  [post_1, post_{2}, ..., post_{t}]
        new_carry = {
            'args': args, 
            'outcomes': outcomes, 
            'beliefs': beliefs, 
            'multiactions': actions,
            'task': task
        }
        return new_carry, action_probs_t

    if data_absence:
        key = prng_key()
        keys = jr.split(key, agent.batch_size)
        outcome_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), task.step(keys)[0])
    else:
        outcome_0 = jtu.tree_map(lambda x: x[..., :1], data['outcomes'])

    init = {
       'args': (agent.D, None,),
       'outcomes': outcome_0, 
       'beliefs': [],
       'multiactions': None, 
       'task': task
    }
    last, multiaction_probs = local_scan(step_fn, init, range(num_trials), axis=1)

    return last, multiaction_probs

def pymdp_likelihood(agent, external_likelihood=None, data=None, task=None, num_blocks=1, num_trials=1, num_agents=1, record_agent=False, **kwargs):
    # Na -> batch dimension - number of different subjects/agents
    # Nb -> number of experimental blocks
    # Nt -> number of trials within each block

    assert num_agents == agent.batch_size

    def step_fn(carry, block_data):
        agent, task = carry
        output, multiaction_probs = pymdp_evolve_trials(agent, block_data, task, num_trials)
        args = output.pop('args')
        multiactions = output.pop('multiactions')
        task = output.pop('task')
        output['beliefs'] = agent.infer_states(
            output['outcomes'],
            args[0],
            past_actions=multiactions,
            qs_hist=args[1],
            mask=None,
        )
        
        deterministic('outcomes', output['outcomes'])
        deterministic('beliefs', output['beliefs'])
        deterministic('multiactions', multiactions)
        deterministic('multiaction_probs', multiaction_probs)
        if task is not None:
            deterministic('states', jtu.tree_map(jnp.stack, task.states))
        
        with plate('num_trials', num_trials):
            with plate('num_agents', num_agents):
                # if block_data contains multiaction_cat field (multiaction as categories) then use that directly 
                # otherwise map multiactions to categories
                if block_data is not None:
                    if 'multiaction_cat' not in block_data:
                        obs = multiaction_to_category(agent.unique_multiactions, multiactions)
                    else:
                        obs = block_data['multiaction_cat']
                else:
                    obs = multiaction_to_category(agent.unique_multiactions, multiactions)
                sample('multiaction_cat', dist.Categorical(probs=multiaction_probs), obs=obs)

                if external_likelihood is not None:
                    external_likelihood(agent, block_data, output, multiaction_probs)

        lr = jnp.ones(agent.batch_size)
        lr_pA = kwargs.pop('lr', lr)
        lr_pA = kwargs.pop('lr_pA', lr)
        lr_pB = kwargs.pop('lr', lr)
        lr_pB = kwargs.pop('lr_pB', lr)
        agent = agent.infer_parameters(output['beliefs'], output['outcomes'], multiactions, lr_pA=lr_pA, lr_pB=lr_pB)

        if record_agent:
            deterministic('agent_sequence', agent)
        
        init_task = task.reset(prng_key()) if task is not None else task
        return (agent, init_task), None
    
    if record_agent:
        deterministic('init_agent', agent)
    
    init_task = task.reset(prng_key()) if task is not None else task
    scan(step_fn, (agent, init_task), data, length=num_blocks)


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


def befit_likelihood(agent, data=None, task=None, num_blocks=1, num_trials=1, num_agents=1, seed=0, **kwargs):
    # num_agents -> batch dimension - number of different subjects/agents
    # blocks -> number of experimental blocks
    # trials -> number of trials within each block

    data_absence = data is None
    if data_absence:
        assert task is not None

    keys = random.split(random.PRNGKey(seed), num_blocks)
    # define prior mean over model parametrs and subjects
    def step_fn(carry, xs):
        init_beliefs = carry
        block_index, key, block_data = xs
        if not data_absence:
            mask = True if block_data['mask'] is None else block_data['mask']
        else:
            mask = True
        
        last_beliefs, logits, beliefs, responses, outcomes, offers = befit_evolve_trials(
            key,
            agent, 
            init_beliefs, 
            block_index, 
            num_trials, 
            data=block_data, 
            task=task
        )

        with plate('runs', num_agents):
            with plate('num_trials', num_trials):
                sample('responses', dist.Categorical(logits=logits).mask(mask), obs=responses)

        deterministic('outcomes', outcomes)
        deterministic('logits', logits)
        deterministic('beliefs', beliefs)
        if offers is not None:
            deterministic('offers', offers)

        return last_beliefs, None

    scan(step_fn, agent.get_beliefs, (jnp.arange(num_blocks), keys, data), length=num_blocks)