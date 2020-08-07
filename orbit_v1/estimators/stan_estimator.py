import logging
import pandas as pd
import multiprocessing
from .base_estimator import BaseEstimator

from ..utils.stan import get_compiled_stan_model
from ..utils.general import update_dict

# todo: add stan docstrings


class StanEstimator(BaseEstimator):
    """Stan Estimator for interaction with PyStan"""
    def __init__(self, num_warmup=900, num_sample=100, chains=4,
                 cores=8, seed=8888, algorithm=None, **kwargs):
        super().__init__()
        self.num_warmup = num_warmup
        self.num_sample = num_sample
        self.chains = chains
        self.cores = cores
        self.seed = seed
        self.algorithm = algorithm

        # init computed configs
        # todo: change to private variables
        self.num_warmup_per_chain = None
        self.num_sample_per_chain = None
        self.num_iter_per_chain = None
        self.total_iter = None

        self._set_computed_stan_configs()

    def _set_computed_stan_configs(self):
        """Sets sampler configs based on init class attributes"""
        # make sure cores can only be as large as the device support
        self.cores = min(self.cores, multiprocessing.cpu_count())
        self.num_warmup_per_chain = int(self.num_warmup/self.chains)
        self.num_sample_per_chain = int(self.num_sample/self.chains)
        self.num_iter_per_chain = self.num_warmup_per_chain + self.num_sample_per_chain
        self.total_iter = self.num_iter_per_chain * self.chains

        if self.verbose:
            msg_template = "Using {} chains, {} cores, {} warmup and {} samples per chain for sampling."
            msg = msg_template.format(
                self.chains, self.cores, self.num_warmup_per_chain, self.num_sample_per_chain)
            logging.info(msg)


class StanEstimatorMCMC(StanEstimator):
    """Stan Estimator for MCMC Sampling"""
    def __init__(self, stan_mcmc_control=None, stan_mcmc_args=None):
        super().__init__()
        self.stan_mcmc_control = stan_mcmc_control
        self.stan_mcmc_args = stan_mcmc_args

        self._set_computed_stan_mcmc_configs()

    def _set_computed_stan_mcmc_configs(self):
        self.stan_mcmc_args = update_dict({}, self.stan_mcmc_args)

    def fit(self, model_payload):
        """Estimate model posteriors with Stan

        Parameters
        ----------
        model_payload : dict
            Model specific payload such as `stan_model_name`, `data_input`,
            `model_param_names`, `stan_init`

        """
        compiled_stan_file = get_compiled_stan_model(model_payload['stan_model_name'])

        stan_mcmc_fit = compiled_stan_file.sampling(
            data=model_payload['data_input'],
            pars=model_payload['model_param_names'],
            iter=self.num_iter_per_chain,
            warmup=self.num_warmup_per_chain,
            chains=self.chains,
            n_jobs=self.cores,
            init=model_payload['stan_init'],
            seed=self.seed,
            algorithm=self.algorithm,
            control=self.stan_mcmc_control,
            **self.stan_mcmc_args
        )

        stan_extract = stan_mcmc_fit.extract(
            pars=model_payload['model_param_names'],
            permuted=False
        )

        # todo: move dimension cleaning function to the model directly
        for key, val in stan_extract.items():
            if len(val.shape) == 2:
                # here `order` is important to make samples flattened by chain
                stan_extract[key] = val.flatten(order='F')
            else:
                stan_extract[key] = val.reshape(-1, val.shape[-1], order='F')

        return stan_extract


class StanEstimatorMAP(StanEstimator):
    """Stan Estimator for MAP Posteriors"""
    pass


class StanEstimatorVI(StanEstimator):
    """Stan Estimator for VI Sampling"""
    pass
