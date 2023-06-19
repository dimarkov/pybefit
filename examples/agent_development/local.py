import torch
import jax.numpy as jnp
from jax import nn, random
from pybefit.agents import Discrete
from pybefit.tasks import Task

class MABTask(Task):
    
    def __init__(self, outcomes):
        blocks, trials, nsub, _ = outcomes.shape
        super().__init__(nsub, blocks, trials)
        self.outcomes = outcomes

    def update_environment(self, block, trial, responses):
        return self.outcomes[block, trial, range(self.nsub), responses]
    
class MABTask(Task):
    def __init__(self, outcomes, backend='torch'):
        blocks, trials, nsub, num_arms = outcomes.shape
        super().__init__(nsub, blocks, trials)
        self.outcomes = outcomes
        self.num_arms = num_arms
        if backend == 'torch':
            self.tensor = torch
            self.one_hot = torch.nn.functional.one_hot
        else:
            self.tensor = jnp
            self.one_hot = nn.one_hot

    def update_environment(self, block, trial, responses):
        return self.tensor.sum(
            self.outcomes[block, trial] * self.one_hot(responses, self.num_arms), -1
        )

class UCBAgent(Discrete):

    def __init__(self, runs=1, blocks=1, trials=1, num_arms=2, backend='torch'):
        # define bernoulli bandit with two outcomes (0, 1) for each arm
        super().__init__(runs, blocks, trials, num_arms, num_arms, 2)
        self.load_backend(backend)


    def load_backend(self, backend):
        self.backend = backend
        if backend == "torch":
            self.tensor = torch
            self.nn = torch.nn.functional
            self.sigmoid = torch.sigmoid

            class Cat(object):
                def __call__(self, logits, *args):
                    dist = torch.distributions.Categorical(logits=logits)
                    return dist.sample()

        else:
            self.tensor = jnp
            self.nn = nn
            self.sigmoid = nn.sigmoid

            class Cat(object):
                def __call__(self, logits, key):
                    return random.categorical(key, logits)

        self.categorical = Cat()

    @property
    def num_params(self):
        return 3
    
    @property
    def get_beliefs(self):
        return (self.q, self.count)
    
    def set_parameters(self, z):
        self.lr = self.sigmoid(4 * z[..., 0] + 2)  # learning rate
        self.c = self.tensor.exp(z[..., 1] - .5)  # exploration strength
        self.beta = self.tensor.exp(z[..., 2] + 2)  # response noise

        batch_shape = z.shape[:-1]
        self.q = self.tensor.zeros(batch_shape + (self.na,))  # q values
        self.count = self.tensor.zeros(batch_shape + (self.na,))  # response count

    def update_beliefs(self, block, trial, response_outcome, **kwargs):
        
        # encode reponses as zero/one array where one is assigned to the chosen arm and zero to all other arms
        beliefs = kwargs.pop('beliefs', (self.q, self.count))
        q = beliefs[0]
        count = beliefs[1]
        response = self.nn.one_hot(response_outcome[0], self.na)

        # add one dimension to the right to outcomes to match dimensionality of responses
        outcome = response_outcome[1][..., None] 


        alpha = self.lr[..., None] / (count + 1)

        # implements self.q[..., response] += alpha * (outcome - self.q[..., response])
        self.q = q + alpha * response * (outcome - q)
        self.count = count + response

        return (self.q, self.count)

    def planning(self, block, trial, **kwargs):

        beliefs = kwargs.pop('beliefs', (self.q, self.count))
        q = beliefs[0]
        count = beliefs[1]

        t = block * self.nt + trial
        logits = q + self.c[..., None] * self.tensor.sqrt( self.tensor.log(t + self.tensor.ones(1))/(count + 1e-4))

        return self.beta[..., None] * logits 
        
    def sample_responses(self, block, trial, logits, **kwargs):

        key = kwargs.pop('key', None)
        return self.categorical(logits, key)