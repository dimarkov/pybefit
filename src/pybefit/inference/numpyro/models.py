from typing import Callable, Dict, Optional

class NumpyroModel(object):

    prior: Callable
    transform: Callable
    likelihood: Callable
    opts: Optional[Dict] = {'prior': {}, 'transform': {}, 'likelihood': {}}

    def __init__(self, prior, transform, likelihood, opts=None) -> None:
        self.prior = prior
        self.transform = transform
        self.likelihood = likelihood
        if opts is not None:
            self.opts = opts

    def __call__(self, data=None):
        z = self.prior(**self.opts['prior'])
        agent = self.transform(z, **self.opts['transform'])
        self.likelihood(agent, data=data, **self.opts['likelihood'])

class NumpyroGuide:
    
    guide: Callable

    def __init__(self, guide) -> None:
        self.guide = guide

    def __call__(self, data=None):
        self.guide(data)