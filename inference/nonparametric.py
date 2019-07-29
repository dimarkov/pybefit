"""This module contains the hierarchical implementations of the parametric model.
"""
from tqdm import tqdm
import pandas as pd

import torch
from torch import zeros, ones
from torch.distributions import constraints, biject_to

import pyro.distributions as dist
from pyro import sample, param, plate, poutine

from pyro import clear_param_store
from pyro.infer import SVI, TraceEnum_ELBO, infer_discrete
from pyro.optim import Adam

from .infer import Inferrer

__all__ = [
    'DirichletProcessPrior'        
]

from torch.nn.functional import pad
def stick_breaking(x):
    return pad(x, (0, 1), value=1) * pad((1 - x).cumprod(-1), (1, 0), value=1)

class DirichletProcessPrior(Inferrer):
    
    def __init__(self, agent, stimulus, responses, mask=None, fixed_params=None, max_components=5):
        self.kmax = max_components
        super(DirichletProcessPrior, self).__init__(agent, stimulus, responses, mask=mask, fixed_params=fixed_params)

    def model(self):
        """
        Generative model of behavior with a hierarchical (horseshoe)
        prior over free model parameters.
        """
        runs = self.runs  # number of independent runs of experiment
        npar = self.npar  # number of parameters
        kmax = self.kmax  # maximal number of components in the mixture model
        
        # parametrise gamma with alpha and rho=alpha/beta
        maa = param('ma_a', ones(1), constraint=constraints.positive)
        mar = param('ma_r', ones(1), constraint=constraints.positive)
        alpha = sample('alpha', dist.Gamma(maa, maa/mar))
        
        beta = sample('beta', dist.Beta(1, alpha).expand([kmax-1]).to_event(1))
        pi = stick_breaking(beta)
    
        ma = param('m_a', torch.ones(kmax, npar), constraint=constraints.positive)
        mr = param('m_r', torch.ones(kmax, npar), constraint=constraints.positive)
        tau = sample('tau', dist.Gamma(ma, ma/mr).to_event(2))
    
        mp = param('m_p', torch.ones(kmax, npar), constraint=constraints.positive)
        loc = sample('loc', dist.Normal(0., 1/torch.sqrt(mp*tau)).to_event(2))
        
        # define prior mean over model parameters and subjects
        with plate('runs', runs):
            d = sample('class', dist.Categorical(pi), infer={"enumerate": "parallel"})
            mu = loc[d]
            std = 1/torch.sqrt(tau[d])
            locs = sample('locs', dist.Normal(mu, std).to_event(1))

        if self.fixed_values:
            x = zeros(runs, self.agent.npar)
            x[:, self.locs['fixed']] = self.values
            x[:, self.locs['free']] = locs
        else:
            x = locs

        self.agent.set_parameters(x)
        
        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                offers = self.stimulus['offers'][b, t]
                self.agent.planning(b, t, offers)
                
                logits = self.agent.logits[-1]
                
                outcomes = self.stimulus['outcomes'][b, t]
                responses = self.responses[b, t]
                
                mask = self.stimulus['mask'][b, t]
                
                self.agent.update_beliefs(b, t, [responses, outcomes], mask=mask)
                
                notnans = self.notnans[b, t]                
                
                if torch.any(notnans):
                    lgs = logits[notnans]
                    res = responses[notnans]
                    with plate('responses_{}_{}'.format(b, t), len(res)):
                        sample('obs_{}_{}'.format(b, t), dist.Categorical(logits=lgs), obs=res)
            
    def guide(self):
        """Approximate posterior for the horseshoe prior. We assume posterior in the form 
        of the multivariate normal distriburtion for the global mean and standard deviation
        and multivariate normal distribution for the parameters of each subject independently.
        """
        nsub = self.runs  # number of subjects
        npar = self.npar  # number of parameters
        kmax = self.kmax  # maximum number of components
        
        gaa = param("ga_a", ones(1), constraint=constraints.positive)
        gar = param("ga_r", .1*ones(1), constraint=constraints.positive)
        alpha = sample('alpha', dist.Gamma(gaa, gaa/gar))
        
        gba = param("gb_beta_a", ones(kmax - 1), constraint=constraints.positive)
        gbb = param("gb_beta_b", ones(kmax - 1), constraint=constraints.positive)
        beta = sample("beta", dist.Beta(gba, gbb).to_event(1))
        
        ga = param("g_a", torch.ones(kmax, npar), constraint=constraints.positive)
        gr = param("g_r", torch.ones(kmax, npar), constraint=constraints.positive)
        tau = sample('tau', dist.Gamma(ga, ga/gr).to_event(2))
        
        gm = param("g_m", .01*torch.randn(kmax, npar))
        gp = param("g_p", torch.ones(kmax, npar), constraint=constraints.positive)
        loc = sample('loc', dist.Normal(gm, 1/torch.sqrt(gp*tau)).to_event(2))
    
#        trns = biject_to(constraints.positive)

#        m_hyp = param('m_hyp', zeros(2*npar))
#        st_hyp = param('scale_tril_hyp', 
#                              torch.eye(2*npar), 
#                              constraint=constraints.lower_cholesky)
#        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, 
#                                                  scale_tril=st_hyp), 
#                            infer={'is_auxiliary': True})
#        
#        unc_mu = hyp[:npar]
#        unc_tau = hyp[npar:]
#    
#        c_tau = trns(unc_tau)
#    
#        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
#        ld_tau = sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)
#    
#        mu = sample("mu", dist.Delta(unc_mu, event_dim=1))
#        tau = sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))
        
        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('scale_tril_locs', torch.eye(npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)
        
        class_probs = param('class_probs', ones(nsub, kmax) / kmax,
                                      constraint=constraints.unit_interval)
        with plate('subjects', nsub):
            d = sample('class', dist.Categorical(class_probs), infer={"enumerate": "parallel"})
            locs = sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
        
        return {'alpha': alpha, 'beta': beta, 'tau': tau, 'loc': loc, 'locs': locs, 'class': d}
    
    def infer_posterior(self, 
                        iter_steps = 1000,
                        num_particles = 10,
                        optim_kwargs = {'lr':.1}):
        """Perform stochastic variational inference over free model parameters.
        """

        clear_param_store()

        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=TraceEnum_ELBO(num_particles=num_particles, max_plate_nesting=1))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
                
        self.loss = loss
            
    def sample_posterior(self, labels, n_samples=1000):
        """Generate samples from posterior distribution.
        """
        nsub = self.runs
        npar = self.npar
        kmax = self.kmax
        
        assert npar == len(labels)
        
        keys = ['locs', 'tau', 'loc', 'alpha', 'beta', 'class']
        
        trans_pars = zeros(n_samples, nsub, npar)
        
        loc_group = zeros(n_samples, kmax, npar)
        tau_group = zeros(n_samples, kmax, npar)
        alpha_group = zeros(n_samples)
        pi_group = zeros(n_samples, kmax)
        
        classes = []
        
        for i in range(n_samples):
            sample = self.guide()
            for key in keys:
                sample.setdefault(key, ones(1))
            
            alpha = sample['alpha']
            beta = sample['beta']
            pi = stick_breaking(beta)
            loc = sample['loc']
            tau = sample['tau']
            locs = sample['locs']
            classes.append(sample['class'])
            
            trans_pars[i] = locs.detach()
            
            loc_group[i] = loc.detach()
            tau_group[i] = tau.detach()
            pi_group[i] = pi.detach()
            alpha_group[i] = alpha.detach()
        
        subject_label = torch.arange(1, nsub+1).repeat(n_samples, 1).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npar).numpy(), columns=labels)
        tp_df['subject'] = subject_label.numpy()
        
        return (tp_df, loc_group, tau_group, pi_group, alpha_group, torch.stack(classes))
    
    def classifier(self, n_samples=1000, temperature=1):
        
        guide_trace = poutine.trace(self.guide).get_trace()
        trained_model = poutine.replay(self.model, trace=guide_trace)  # replay the globals
        
        classes = []
        for n in range(n_samples):
            inferred_model = infer_discrete(trained_model, temperature=temperature, first_available_dim=-1)  # avoid conflict with data plate
            trace = poutine.trace(inferred_model).get_trace()
            classes.append(trace.nodes["class"]["value"])
            
        self.classes = torch.stack(classes)
        
        return self.classes
        
    
    def _get_quantiles(self, quantiles):
        """
        Returns posterior quantiles each latent variable.

        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.Tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        
        self.means = [param('m_locs'), param('m_hyp')]
        self.stds = [param('scale_tril_hyp'), param('scale_tril_locs')]
        
        quantiles = torch.tensor(quantiles).reshape(1, 3)

        
        m_locs = param('m_locs').reshape(-1, 1)
        s_locs = param('scale_tril_locs').diagonal(dim1=-2, dim2=-1).reshape(-1, 1)
        
        latents = dist.Normal(m_locs, s_locs).icdf(quantiles).reshape(self.runs, -1, 3)
        result = {'locs': latents}

#        m_hyp = param('m_hyp').reshape(-1, 1)
#        s_hyp = param('scale_tril_hyp').diagonal(dim1=-2, dim2=-1).reshape(-1, 1)
#        
#        latents = dist.Normal(m_hyp, s_hyp).icdf(quantiles).reshape(-1, 1)
#        
#        result['mu'] = latents[:self.npar]
#        result['tau'] = latents[self.npar:].exp()
                
        return result