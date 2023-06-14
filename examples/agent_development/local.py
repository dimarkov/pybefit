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
                def __init__(self, seed=0):
                    torch.manual_seed(seed)

                def __call__(self, logits):
                    dist = torch.distributions.Categorical(logits=logits)
                    return dist.sample()

        else:
            self.tensor = jnp
            self.nn = nn
            self.sigmoid = nn.sigmoid

            class Cat(object):
                def __init__(self, seed=0):
                    self.key = random.PRNGKey(seed)

                def __call__(self, logits):
                    self.key, key = random.split(self.key)
                    return random.categorical(key, logits)

        self.categorical = Cat(seed=0)


    @property
    def num_params(self):
        return 3
    
    def set_parameters(self, z):
        self.lr = self.sigmoid(z[..., 0] - 1)  # learning rate
        self.c = self.tensor.exp(0.1 * z[..., 1] - 1)  # exploration strength
        self.beta = self.tensor.exp(0.1 * z[..., 2] + 2)  # response noise

        batch_shape = z.shape[:-1]
        self.q = self.tensor.zeros(batch_shape + (self.na,))  # q values
        self.count = self.tensor.zeros(batch_shape + (self.na,))  # response count

    def update_beliefs(self, block, trial, response_outcome, **kwargs):
        
        # encode reponses as zero/one array where one is assigned to the chosen arm and zero to all other arms
        response = self.nn.one_hot(response_outcome[0], self.na)

        # add one dimension to the right to outcomes to match dimensionality of responses
        outcome = response_outcome[1][..., None] 


        alpha = self.lr[..., None] / (self.count + 1)

        # implements self.q[..., response] += alpha * (outcome - self.q[..., response])
        self.q += alpha * response * (outcome - self.q)
        self.count += response

    def planning(self, block, trial, *args, **kwargs):
        
        logits = self.q + self.c[..., None] * self.tensor.sqrt( self.tensor.log(trial + self.tensor.ones(1))/(self.count + 1e-6) )

        return self.beta[..., None] * logits 
        
    def sample_responses(self, block, trial, logits, *args, **kwargs):

        return self.categorical(logits)