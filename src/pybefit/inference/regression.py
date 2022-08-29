import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist


from jax import lax, vmap
from jax.nn import logsumexp
from numpyro.distributions import constraints
from numpyro import handlers, sample, deterministic, plate
from numpyro.infer import MCMC, NUTS, Predictive, init_to_feasible, log_likelihood
from numpyro.infer.reparam import TransformReparam
from numpyro.distributions.transforms import Transform, AffineTransform, ComposeTransform

class QRTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, R, R_inv):
        self.R = R
        self.R_inv = R_inv

    def __call__(self, x):
        return jnp.squeeze(
            jnp.matmul(self.R, x[..., jnp.newaxis]), axis=-1
        )

    def _inverse(self, y):
        return jnp.squeeze(
            jnp.matmul(self.R_inv, y[..., jnp.newaxis]), axis=-1
        )

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.broadcast_to(
            jnp.log(jnp.diagonal(self.R, axis1=-2, axis2=-1)).sum(-1),
            jnp.shape(x)[:-1],
        )

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.R.shape[:-1])

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return lax.broadcast_shapes(shape, self.R.shape[:-1])

class BayesRegression(object):
    def __init__(self, rng_key, X, tau=1., p0=None, fixed_lam=False, with_qr=True, reg_type='linear'):
        self.batch, self.N, self.D = X.shape
        self.X = X
        self.rng_key = rng_key
        self.tau = tau
        self.fixed_lam = fixed_lam
        self.with_qr = with_qr
        self.p0 = p0
        self.type = reg_type # type of the rergression problem
        
        if self.with_qr:
            # use QR decomposition
            self.Q, self.R = vmap(jnp.linalg.qr)(X)
            self.R_inv = vmap(jnp.linalg.inv)(self.R)
        
    def model(self, obs=None):

        if self.p0 is None:
            if self.batch > 1:
                a = sample('a', dist.InverseGamma(1/2, 1/2))
                b = sample('b', dist.InverseGamma(1/2, 1/2))

                rho = sample('rho', dist.Beta(1., 1.))
                p0 = deterministic('p0', 1 + (self.D - 1) * rho)
            else:
                a = 2.
                b = 2.
        else:
            p0 = self.p0
               
        with plate('batch', self.batch):
            sigma_sqr = sample('var_sigma^2', dist.InverseGamma(a, 1.))
            sigma = deterministic('sigma', jnp.sqrt(b * sigma_sqr))
            if self.p0 is None:
                if self.batch > 1:
                    tau0 = p0 * sigma / ((self.D + 1 - p0) * jnp.sqrt(self.N))
                    tau = sample('tau', dist.HalfCauchy(tau0))
                else:
                    tau = self.tau * sigma
            else:
                tau0 = p0 * sigma / ((self.D + 1 - p0) * jnp.sqrt(self.N))
                tau = sample('tau', dist.HalfCauchy(tau0))
                
            if self.fixed_lam:
                lam = deterministic('lam', jnp.ones(self.D))
            else:
                lam = sample('lam', dist.HalfCauchy(1.).expand([self.D]).to_event(1))

            if self.with_qr:
                rt = QRTransform(self.R, self.R_inv)
                aff = AffineTransform(0., jnp.expand_dims(tau, -1) * lam)
                ct = ComposeTransform([aff, rt])
                with handlers.reparam(config={"theta": TransformReparam()}):
                    theta = sample(
                        'theta', 
                        dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.D]).to_event(1), ct)
                    )

                deterministic('beta', rt.inv(theta))
                tmp = jnp.squeeze(jnp.matmul(self.Q, theta[..., jnp.newaxis]), axis=-1)

            else:
                aff = AffineTransform(0., jnp.expand_dims(tau, -1) * lam)
                with handlers.reparam(config={"beta": TransformReparam()}):
                    beta = sample(
                        'beta', 
                        dist.TransformedDistribution(dist.Normal(0., 1.).expand([self.D]).to_event(1), aff)
                    )
                tmp = jnp.squeeze(jnp.matmul(self.X, beta[..., jnp.newaxis]), axis=-1)

            alpha = sample('alpha', dist.Normal(0., 10.))
            mu = deterministic('mu', jnp.expand_dims(alpha, -1) + tmp)
            
            with plate('data', self.N):
                if self.type == 'linear':
                    sample('obs', dist.Normal(mu.T, sigma), obs=obs)
                elif self.type == 'logistic':
                    sample('obs', dist.Bernoulli(logits=mu.T), obs=obs)
                elif self.type == 'possion':
                    sample('obs', dist.Poisson(jnp.exp(mu.T)), obs=obs)
    
    def fit(self, data, num_samples=1000, warmup_steps=1000, num_chains=1, summary=True, progress_bar=True):
        self.rng_key, _rng_key = random.split(self.rng_key)

        nuts_kernel = NUTS(self.model, init_strategy=init_to_feasible)
        mcmc = MCMC(nuts_kernel, 
                    num_warmup=warmup_steps, 
                    num_samples=num_samples, 
                    num_chains=num_chains,
                    chain_method='vectorized',
                    progress_bar=progress_bar)
        mcmc.run(_rng_key, obs=data)

        if summary:
            mcmc.print_summary()

        samples = mcmc.get_samples(group_by_chain=False)
        self.mcmc = mcmc
        self.samples = samples

        return samples

    def predictions(self, rng_key, num_samples=1):
        predictive = Predictive(self.model, self.samples, num_samples=num_samples)

        return predictive.get_samples(rng_key)['obs']

    def waic(self, data):
        '''Widely applicable information criterion: 
        Watanabe, Sumio. "A widely applicable Bayesian information criterion." Journal of Machine Learning Research 14.27 (2013): 867-897.
        '''
        log_lk = log_likelihood(self.model, self.samples, obs=data)['obs']
        
        S = log_lk.shape[0]

        T = logsumexp(log_lk, 0) - jnp.log(S)
        V = jnp.square(log_lk).mean(0) - jnp.square(log_lk.mean(0))

        return - T.mean(0) + V.mean(0)