from abc import abstractmethod

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random

from .base_estimator import BaseEstimator
from ..utils.pyro import get_pyro_model


class NumPyroEstimator(BaseEstimator):
    """Abstract PyroEstimator with shared args for all PyroEstimator child classes

    Parameters
    ----------
    num_sample : int
        Number of estimator steps in optimization
    num_warmup : int
        Estimator learning rate
    Seed int
    message :  int
        Print to console every `message` number of steps
    kwargs
        Additional BaseEstimator args
    """
    def __init__(self, num_warmup=1000, num_sample=1000, **kwargs):
        super().__init__(**kwargs)
        self.num_warmup = num_warmup
        self.num_sample = num_sample

    @abstractmethod
    def fit(self, model_name, model_param_names, data_input, fitter=None, init_values=None):
        # fitter as abstract
        if fitter is None:
            fitter = get_pyro_model(model_name, is_num_pyro=True)
        # concrete
        model = fitter(data_input)
        nuts_kernel = NUTS(model)
        numpyro.set_host_device_count(4)
        mcmc = MCMC(nuts_kernel, num_samples=self.num_sample, num_warmup=self.num_warmup, num_chains=4)
        rng_key = random.PRNGKey(self.seed)
        mcmc.run(rng_key)
        posteriors = mcmc.get_samples()


        return posteriors
