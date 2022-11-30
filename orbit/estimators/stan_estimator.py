from abc import abstractmethod
from copy import copy
import logging
import multiprocessing
from sys import platform, version_info

from .base_estimator import BaseEstimator
from ..exceptions import EstimatorException
from ..utils.stan import get_compiled_stan_model, suppress_stdout_stderr
from ..utils.general import update_dict

if platform == "darwin" and version_info[0] == 3 and version_info[1] >= 8:
    # fix issue in Python 3.9
    multiprocessing.set_start_method("fork", force=True)
logger = logging.getLogger("orbit")


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_warmup = num_warmup
        self.num_sample = num_sample
        self.chains = chains
        self.cores = cores
        self.algorithm = algorithm

        # stan_init fallback if not provided in model
        # this arg is passed in through `model_payload` in `fit()` to override
        self.stan_init = "random"

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
    stan_mcmc_control : dict
        Supplemental stan control parameters to pass to PyStan.sampling()
    stan_mcmc_args : dict
        Supplemental stan mcmc args to pass to PyStan.sampling()

    """

    def __init__(self, stan_mcmc_control=None, stan_mcmc_args=None, **kwargs):
        super().__init__(**kwargs)
        self.stan_mcmc_control = stan_mcmc_control
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
        compiled_stan_file = get_compiled_stan_model(model_name)

        #   passing callable from the model as seen in `initfun1()`
        #   https://pystan2.readthedocs.io/en/latest/api.html
        #   if None, use default as defined in class variable
        init_values = init_values or self.stan_init

        # T_STAR is used as sampling temperature which is used for WBIC calculation
        data_input.update({"T_STAR": sampling_temperature})

        if self.verbose:
            msg_template = (
                "Sampling (PyStan) with chains: {:d}, cores: {:d}, temperature: {:.3f}, "
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

        # with io.capture_output() as captured:
        with suppress_stdout_stderr():
            stan_mcmc_fit = compiled_stan_file.sampling(
                data=data_input,
                pars=model_param_names + ["loglk"],
                iter=self._num_iter_per_chain,
                warmup=self._num_warmup_per_chain,
                chains=self.chains,
                n_jobs=self.cores,
                # fall back to default if not provided by model payload
                init=init_values,
                seed=self.seed,
                algorithm=self.algorithm,
                control=self.stan_mcmc_control,
                **self._stan_mcmc_args,
            )

        posteriors = stan_mcmc_fit.extract(pars=model_param_names, permuted=False)

        # todo: move dimension cleaning function to the model directly
        # flatten the first two dims by preserving the chain order
        for key, val in posteriors.items():
            if len(val.shape) == 2:
                # here `order` is important to make samples flattened by chain
                posteriors[key] = val.flatten(order="F")
            else:
                posteriors[key] = val.reshape((-1, *val.shape[2:]), order="F")

        # extract `log_prob` in addition to defined model params
        # to make naming consistent across api; we move lp along with warm up lp to `training_metrics`
        # model_param_names_with_lp = model_param_names[:] + ['lp__']
        loglk = stan_mcmc_fit.extract(pars=["loglk"], permuted=True)["loglk"]
        training_metrics = {"loglk": loglk}
        training_metrics.update(
            {"log_posterior": stan_mcmc_fit.get_logposterior(inc_warmup=True)}
        )
        training_metrics.update({"sampling_temperature": sampling_temperature})

        return posteriors, training_metrics


class StanEstimatorMAP(StanEstimator):
    """Stan Estimator for MAP Posteriors

    Parameters
    ----------
    stan_map_args : dict
        Supplemental stan vi args to pass to PyStan.optimizing()

    """

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
            msg_template = "Optimizing (PyStan) with algorithm: {}."
            if self.algorithm is None:
                algorithm = "LBFGS"
            else:
                algorithm = self.algorithm
            msg = msg_template.format(algorithm)
            logger.info(msg)

    def fit(
        self, model_name, model_param_names, data_input, fitter=None, init_values=None
    ):
        compiled_stan_file = get_compiled_stan_model(model_name)
        data_input.update({"T_STAR": 1.0})

        # passing callable from the model as seen in `initfun1()`
        init_values = init_values or self.stan_init

        # in case optimizing fails with given algorithm fallback to `Newton`
        try:
            with suppress_stdout_stderr():
                stan_extract = compiled_stan_file.optimizing(
                    data=data_input,
                    init=init_values,
                    seed=self.seed,
                    algorithm=self.algorithm,
                    **self._stan_map_args,
                )
        except RuntimeError:
            self.algorithm = "Newton"
            with suppress_stdout_stderr():
                stan_extract = compiled_stan_file.optimizing(
                    data=data_input,
                    init=init_values,
                    seed=self.seed,
                    algorithm=self.algorithm,
                    **self._stan_map_args,
                )

        # make sure that model param names are a subset of stan extract keys
        invalid_model_param = set(model_param_names) - set(list(stan_extract.keys()))
        if invalid_model_param:
            raise EstimatorException(
                "Stan model definition does not contain required parameters"
            )

        # `stan.optimizing` automatically returns all defined parameters
        # filter out unnecessary keys
        posteriors = {param: stan_extract[param] for param in model_param_names}
        training_metrics = dict()

        # extract `log_prob` in addition to defined model params
        # this is for the BIC calculation
        training_metrics.update({"loglk": stan_extract["loglk"]})
        # FIXME: this needs to be the full length of all parameters instead of the one we sampled?
        # FIXME: or it should be not include latent varaibles / derive variables?
        training_metrics.update({"num_of_params": len(model_param_names)})

        return posteriors, training_metrics
