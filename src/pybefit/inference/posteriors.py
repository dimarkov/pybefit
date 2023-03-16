
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

from .priors import ModelBase


class NormalPosterior(ModelBase):
    """A flat Normal prior over free model parameters.
    """
    
    def __init__(self, num_params, num_agents, backend=BACKEND):
        super().__init__(num_params=num_params, num_agents=num_agents, backend=backend)

    def __call__(self, *args, **kwargs):
        na = self.num_agents
        np = self.num_params
        
        # posterior distribution over subject specific model parameters
        with self.plate('agents', na):
            loc = self.param('loc', self.tensor.zeros((na, np)))
            scale_tril = self.param('scale_tril', self.tensor.broadcast_to(self.init_scale * self.tensor.eye(np), (na, np, np)), constraint=self.constraints.softplus_lower_cholesky)
            z = self.sample('z', self.dist.MultivariateNormal(loc, scale_tril=scale_tril))

        return z


class NormalGammaPosterior(ModelBase):
    """NormalGamma over free model parameters.
    """
    def __init__(self, num_params, num_agents, init_scale=.1, backend=BACKEND):
        super().__init__(num_params=num_params, num_agents=num_agents, backend=backend)
        self.init_scale = init_scale

    def __call__(self, *args, **kwargs):
        na = self.num_agents
        np = self.num_params

        # define hyper priors over model parameters
        loc_tau = self.param('loc.tau', self.tensor.zeros(np))
        scale_tau = self.param('scale.tau', self.init_scale * self.tensor.ones(np), constraint=self.constraints.softplus_positive)
        tau = self.sample('var_tau', self.dist.LogNormal(loc_tau, scale_tau).to_event(1))

        sigma = self.tensor.sqrt(1/tau)

        # each model parameter has a hyperprior defining group level mean
        loc = self.param('loc.mu', self.tensor.zeros(np)) / sigma
        scale = self.param('scale.mu', self.init_scale * self.tensor.ones(np), constraint=self.constraints.softplus_positive)
        self.sample('mu_decentered', self.dist.Normal(loc, scale).to_event(1))

        # posterior distirbution over model parametrs for individual subjects
        with self.plate('agents', na):
            loc = self.param('loc', self.tensor.zeros((na, np))) / sigma
            scale_tril = self.param('scale_tril', self.tensor.broadcast_to(self.init_scale * self.tensor.eye(np), (na, np, np)), constraint=self.constraints.softplus_lower_cholesky)
            self.sample('z_decentered', self.dist.MultivariateNormal(loc, scale_tril=scale_tril))


class RegularisedHorseshoePosterior(ModelBase):
    """Posterior distribution for the regularised horseshoe prior.
    """

    def __init__(self, num_params, num_agents, init_scale=.1, backend=BACKEND):
        super().__init__(num_params=num_params, num_agents=num_agents, backend=backend)
        self.init_scale = init_scale

    def __call__(self, *args, **kwargs):
        na = self.num_agents
        np = self.num_params

        # define prior uncertanty over model parameters
        loc_c = self.param('loc.c', self.tensor.zeros(1))
        scale_c = self.param('scale.c', self.init_scale * self.tensor.ones(1), constraint=self.constraints.softplus_positive)
        c_sqr_inv = self.sample('c^{-2}', self.dist.LogNormal(loc_c, scale_c).to_event(1))

        loc_u = self.param('loc.u', self.tensor.zeros(np + 1))
        scale_u = self.param('scale.u', self.init_scale * self.tensor.ones(np + 1), constraint=self.constraints.softplus_positive)
        u = self.sample('u', self.dist.LogNormal(loc_u, scale_u).to_event(1))

        loc_v = self.param('loc.v', self.tensor.zeros(np + 1))
        scale_v = self.param('scale.v', self.init_scale * self.tensor.ones(np + 1), constraint=self.constraints.softplus_positive)
        v = self.sample('v', self.dist.LogNormal(loc_v, scale_v).to_event(1))

        psi = u[..., -1:] * u[..., :-1]
        ksi = v[..., -1:] * v[..., :-1]

        sigma = self.tensor.sqrt( psi/(ksi + c_sqr_inv * psi ))

        # each model parameter has a hyperparameter defining a group level mean
        loc_mu = self.param('loc.mu', self.tensor.zeros(np))
        scale_mu = self.param('scale.mu', self.init_scale * self.tensor.ones(np), constraint=self.constraints.softplus_positive)
        self.sample('mu', self.dist.Normal(loc_mu, scale_mu).to_event(1))

        # posterior distirbution over model parametrs for individual subjects
        with self.plate('agents', na):
            loc = self.param('loc', self.tensor.zeros((na, np))) / sigma
            scale_tril = self.param('scale_tril', self.tensor.broadcast_to(self.init_scale * self.tensor.eye(np), (na, np, np)), constraint=self.constraints.softplus_lower_cholesky)
            z = self.sample('z_decentered', self.dist.MultivariateNormal(loc, scale_tril=scale_tril))