try:
    import numpyro as npyro
    import numpyro.distributions as ndist
    from numpyro import handlers
    from numpyro.infer import reparam as npyro_reparam
    
    BACKEND = 'numpyro'  # default backend

except:
    print('Numpyro not installed, trying to import Pyro')
    try:
        import pyro
        BACKEND = 'pyro'
    except:
        print('Please install either Pyro or Numpyro PPLs')

else:
    import jax.numpy as jnp

finally:
    from typing import Optional, Dict
    try:
        import pyro
        import torch
        import pyro.distributions as pdist
        from pyro import poutine
        from pyro.infer import reparam as pyro_reparam

    except:
        pass


class ModelBase(object):
    num_params: int  # number of free model parameters
    num_agents: int  # number of independent agents/subjects performing the experiment
    backend: str

    def __init__(self, num_params, num_agents, backend=BACKEND):
        self.num_params = num_params
        self.num_agents = num_agents
        self.backend = backend

        self.__set_backend()
    
    def __set_backend(self):
        if self.backend == 'pyro':
            self.tensor = torch
            self.dist = pdist
            self.constraints = pdist.constraints
            self.plate = pyro.plate
            self.sample = pyro.sample
            self.param = pyro.param
            self.deterministic = pyro.deterministic
            self.func_reparam = poutine.reparam
            self.reparam = pyro_reparam

        elif self.backend == 'numpyro':
            self.tensor = jnp
            self.dist = ndist
            self.constraints = ndist.constraints
            self.plate = npyro.plate
            self.sample = npyro.sample
            self.param = npyro.param
            self.deterministic = npyro.deterministic
            self.reparam = npyro_reparam
            self.func_reparam = handlers.reparam

        else:
            raise NotImplementedError


class Normal(ModelBase):
    """A flat Normal prior over free model parameters.
    """
    
    def __init__(self, num_params, num_agents, init_scale=1., backend=BACKEND):
        super().__init__(num_params=num_params, num_agents=num_agents, backend=backend)
        self.init_scale = init_scale

    def __call__(self, *args, **kwargs):
        na = self.num_agents
        np = self.num_params
        
        # each model parameter has a hyper-parameters defining group level mean and uncertainty
        m = self.tensor.zeros(np)
        s = self.init_scale * self.tensor.ones(np)
        
        # parameters for individual agents are sampled from the same prior
        with self.plate('agents', na):
            z = self.sample('z', self.dist.Normal(m, s).to_event(1))

        return z


class NormalGamma(ModelBase):
    """NormalGamma over free model parameters.
    """
    def __init__(self, num_params, num_agents, init_scale=1., backend=BACKEND):
        super().__init__(num_params=num_params, num_agents=num_agents, backend=backend)
        self.init_scale = init_scale

    def __call__(self, *args, **kwargs):
        na = self.num_agents
        np = self.num_params

        # define hyper priors over model parameters
        a = 2 * self.tensor.ones(np)
        b = 2 * self.tensor.ones(np)
        tau = self.sample('var_tau', self.dist.Gamma(a, b).to_event(1))

        # prior uncertainty is sampled from inverse gamma distribution for each parameter
        sigma = self.deterministic('sigma', self.tensor.sqrt(1/tau)) 

        # normal-gamma prior has two hyperparameters which are jointly optimized with 
        # the parameters of the approximate posterior in the case of stochastic variational inference
        m = self.tensor.zeros(np)
        s = self.init_scale * self.tensor.ones(np)
        
        # each model parameter has a hyperprior defining group level mean
        with self.func_reparam(config={"mu": self.reparam.LocScaleReparam(0.)}):
            mu = self.sample('mu', self.dist.Normal(m, s*sigma).to_event(1))

        # parameters for individual agents are sampled from the same prior
        with self.plate('agents', na):
            with self.func_reparam(config={"z": self.reparam.LocScaleReparam(0.)}):
                z = self.sample('z', self.dist.Normal(mu, sigma).to_event(1))

        return z


class RegularisedHorseshoe(ModelBase):
    """Regularised horseshoe prior over free model parameters. For details see: Piironen, Juho, and Aki Vehtari. "Sparsity information and regularization in the horseshoe and other shrinkage priors." (2017): 5018-5051.
    """

    def __init__(self, num_params, num_agents, backend=BACKEND):
        super().__init__(num_params=num_params, num_agents=num_agents, backend=backend)

    def __call__(self, *args, **kwargs):
        na = self.num_agents
        np = self.num_params

        # each model parameter has a hyperparameter defining a group level mean
        mu = self.sample('mu', self.dist.Normal(self.tensor.zeros(np), 10 * self.tensor.ones(np)).to_event(1))

        # define prior uncertanty over model parameters
        c_sqr_inv = self.sample('c^{-2}', self.dist.Gamma(2 * self.tensor.ones(1), 2 * self.tensor.ones(1)).to_event(1))

        u = self.sample('u', self.dist.Gamma(self.tensor.ones(np + 1)/2, self.tensor.ones(np + 1)).to_event(1))
        v = self.sample('v', self.dist.Gamma(self.tensor.ones(np + 1)/2, self.tensor.ones(np + 1)).to_event(1))

        psi = u[..., -1:] * u[..., :-1]
        ksi = v[..., -1:] * v[..., :-1]

        sigma = self.deterministic('sigma', self.tensor.sqrt( psi/(ksi + c_sqr_inv * psi )))

        # define prior mean over model parametrs and subjects
        with self.plate('agents', na):
            with self.func_reparam(config={"z": self.reparam.LocScaleReparam(0.)}):
                z = self.sample('z', self.dist.Normal(mu, sigma).to_event(1))

        return z