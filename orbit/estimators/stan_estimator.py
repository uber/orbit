from abc import abstractmethod
from copy import copy
import logging
import multiprocessing
from sys import platform, version_info

from .base_estimator import BaseEstimator
from ..exceptions import EstimatorException
from ..utils.stan import get_compiled_stan_model, suppress_stdout_stderr
from ..utils.general import update_dict
from ..utils.logger import get_logger
from ..utils.set_cmdstan_path import set_cmdstan_path

logger = get_logger("orbit")

# Make sure models are using the right cmdstan folder
# set_cmdstan_path()
   
class StanEstimator(BaseEstimator):
    """Abstract StanEstimator with shared args for all StanEstimator child classes

    Parameters
    ----------
    num_warmup : int
        Number of samples to warm up and to be discarded, default 900
    num_sample : int
        Number of samples to return, default 100
    chains : int
        Number of chains in stan sampler, default 4
    cores : int
        Number of cores for parallel processing, default max(cores, multiprocessing.cpu_count())
    algorithm : str
        If None, default to Stan defaults
    suppress_stan_log : bool
        If False, turn off cmdstanpy logger. Default as False.
    kwargs
        Additional `BaseEstimator` class args

    """

    def __init__(
        self,
        num_warmup=900,
        num_sample=100,
        chains=4,
        cores=8,
        algorithm=None,
        suppress_stan_log=True,
        **kwargs,
    ):
        # see https://mc-stan.org/cmdstanpy/users-guide/outputs.html for details
        cmdstanpy_logger = logging.getLogger("cmdstanpy")
        cmdstanpy_logger.disabled = suppress_stan_log

        super().__init__(**kwargs)
        self.num_warmup = num_warmup
        self.num_sample = num_sample
        self.chains = chains
        self.cores = cores
        self.algorithm = algorithm

        # stan_init fallback if not provided in model
        # this arg is passed in through `model_payload` in `fit()` to override
        # self.stan_init = "random"

        # init computed configs
        self._num_warmup_per_chain = None
        self._num_sample_per_chain = None
        self._num_iter_per_chain = None
        self._total_iter = None

        self._set_computed_stan_configs()

    def _set_computed_stan_configs(self):
        """Sets sampler configs based on init class attributes"""
        # make sure cores can only be as large as the device support
        self.cores = min(self.cores, multiprocessing.cpu_count())
        self._num_warmup_per_chain = int(self.num_warmup / self.chains)
        self._num_sample_per_chain = int(self.num_sample / self.chains)
        self._num_iter_per_chain = (
            self._num_warmup_per_chain + self._num_sample_per_chain
        )
        self._total_iter = self._num_iter_per_chain * self.chains

    @abstractmethod
    def fit(
        self, model_name, model_param_names, data_input, fitter=None, init_values=None
    ):
        raise NotImplementedError("Concrete fit() method must be implemented")


class StanEstimatorMCMC(StanEstimator):
    """Stan Estimator for MCMC Sampling

    Parameters
    ----------
    stan_mcmc_args : dict
        Supplemental stan mcmc args to pass to CmdStandPy.sampling()

    """

    def __init__(self, stan_mcmc_args=None, **kwargs):
        super().__init__(**kwargs)
        self.stan_mcmc_args = stan_mcmc_args

        # init computed args
        self._stan_mcmc_args = copy(self.stan_mcmc_args)
        self._set_computed_stan_mcmc_configs()

    def _set_computed_stan_mcmc_configs(self):
        self._stan_mcmc_args = update_dict({}, self._stan_mcmc_args)

    def fit(
        self,
        model_name,
        model_param_names,
        sampling_temperature,
        data_input,
        fitter=None,
        init_values=None,
    ):
        # T_STAR is used as sampling temperature which is used for WBIC calculation
        data_input.update({"T_STAR": sampling_temperature})
        if self.verbose:
            msg_template = (
                "Sampling (CmdStanPy) with chains: {:d}, cores: {:d}, temperature: {:.3f}, "
                "warmups (per chain): {:d} and samples(per chain): {:d}."
            )
            msg = msg_template.format(
                self.chains,
                self.cores,
                sampling_temperature,
                self._num_warmup_per_chain,
                self._num_sample_per_chain,
            )
            logger.info(msg)

        compiled_mod = get_compiled_stan_model(model_name)

        # check https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.sample
        # for additional args
        stan_mcmc_fit = compiled_mod.sample(
            data=data_input,
            iter_sampling=self._num_sample_per_chain,
            iter_warmup=self._num_warmup_per_chain,
            chains=self.chains,
            parallel_chains=self.cores,
            inits=init_values,
            seed=self.seed,
            **self._stan_mcmc_args,
        )

        stan_extract = stan_mcmc_fit.stan_variables()
        posteriors = {
            param: stan_extract[param] for param in model_param_names + ["loglk"]
        }

        training_metrics = {
            "loglk": posteriors["loglk"],
            "sampling_temperature": sampling_temperature,
        }

        return posteriors, training_metrics


class StanEstimatorMAP(StanEstimator):
    """Stan Estimator for MAP Posteriors"""

    def __init__(self, stan_map_args=None, **kwargs):
        super().__init__(**kwargs)
        self.stan_map_args = stan_map_args

        # init computed args
        self._stan_map_args = copy(self.stan_map_args)

        # set defaults
        self._set_computed_stan_map_configs()

    def _set_computed_stan_map_configs(self):
        default_stan_map_args = {}
        self._stan_map_args = update_dict(default_stan_map_args, self._stan_map_args)
        # TODO: should it be within the fit block?
        if self.verbose:
            msg_template = "Optimizing (CmdStanPy) with algorithm: {}."
            if self.algorithm is None:
                algorithm = "LBFGS"
            else:
                algorithm = self.algorithm
            msg = msg_template.format(algorithm)
            logger.info(msg)

    def fit(
        self,
        model_name,
        model_param_names,
        data_input,
        fitter=None,
        init_values=None,
    ):
        compiled_mod = get_compiled_stan_model(model_name)
        data_input.update({"T_STAR": 1.0})

        # passing callable from the model as seen in `initfun1()`
        # init_values = init_values or self.stan_init

        # in case optimizing fails with given algorithm fallback to `Newton`
        # init values interface can be referred here: https://cmdstanpy.readthedocs.io/en/stable-0.9.65/api.html
        # Dict [Str, np.array] where key is the param name and value array dim matches the param dim
        try:
            stan_fit = compiled_mod.optimize(
                data=data_input,
                inits=init_values,
                seed=self.seed,
                algorithm=self.algorithm,
                **self._stan_map_args,
            )
        except RuntimeError:
            self.algorithm = "Newton"
            stan_fit = compiled_mod.optimize(
                data=data_input,
                inits=init_values,
                seed=self.seed,
                algorithm=self.algorithm,
                **self._stan_map_args,
            )

        stan_extract = stan_fit.stan_variables()
        # make sure that model param names are a subset of stan extract keys
        invalid_model_param = set(model_param_names) - set(list(stan_extract.keys()))
        if invalid_model_param:
            raise EstimatorException(
                "Stan model definition does not contain required parameters"
            )

        # `stan.optimizing` automatically returns all defined parameters
        # filter out unnecessary keys
        posteriors = {
            param: stan_extract[param] for param in model_param_names + ["loglk"]
        }

        training_metrics = {
            # loglk is needed for BIC calculation
            "loglk": stan_extract["loglk"],
            # TODO: this needs to be the full length of all parameters instead of the one we sampled?
            # TODO: i.e. should it include latent variables / derive variables?
            "num_of_params": len(model_param_names),
        }

        return posteriors, training_metrics
