from abc import abstractmethod
from collections import OrderedDict
import logging
import numpy as np
from copy import copy
import multiprocessing
from .base_estimator import BaseEstimator
from ..exceptions import EstimatorException
from ..utils.stan import get_compiled_stan_model
from ..utils.general import update_dict

# todo: add stan docstrings


class StanEstimator(BaseEstimator):
    """Abstract StanEstimator with shared args for all StanEstimator child classes

    Parameters
    ----------
    num_warmup : int
        Number of samples to discard, default 900
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
    def __init__(self, num_warmup=900, num_sample=100, chains=4,
                 cores=8, algorithm=None, **kwargs):
        super().__init__(**kwargs)
        self.num_warmup = num_warmup
        self.num_sample = num_sample
        self.chains = chains
        self.cores = cores
        self.algorithm = algorithm

        # stan_init fallback if not provided in model
        # this arg is passed in through `model_payload` in `fit()` to override
        self.stan_init = 'random'

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
        self._num_iter_per_chain = self._num_warmup_per_chain + self._num_sample_per_chain
        self._total_iter = self._num_iter_per_chain * self.chains

        if self.verbose:
            msg_template = "Using {} chains, {} cores, {} warmup and {} samples per chain for sampling."
            msg = msg_template.format(
                self.chains, self.cores, self._num_warmup_per_chain, self._num_sample_per_chain)
            logging.info(msg)

    @abstractmethod
    def fit(self, model_name, model_param_names, data_input, fitter=None, init_values=None):
        raise NotImplementedError('Concrete fit() method must be implemented')


class StanEstimatorMCMC(StanEstimator):
    """Stan Estimator for MCMC Sampling

    Parameters
    ----------
    stan_mcmc_control : dict
        Supplemental stan control parameters to pass to PyStan.sampling()
    stan_mcmc_args : dict
        Supplemental stan mcmc args to pass to PyStan.sampling()

    """
    # is_mcmc boolean indicator -- some models are parameterized slightly different for
    # MCMC estimator vs other estimators for convergence. Indicator let's model and estimator
    # to remain independent
    _is_mcmc_estimator = True

    def __init__(self, stan_mcmc_control=None, stan_mcmc_args=None, **kwargs):
        super().__init__(**kwargs)
        self.stan_mcmc_control = stan_mcmc_control
        self.stan_mcmc_args = stan_mcmc_args

        # init computed args
        self._stan_mcmc_args = copy(self.stan_mcmc_args)

        self._set_computed_stan_mcmc_configs()

    def _set_computed_stan_mcmc_configs(self):
        self._stan_mcmc_args = update_dict({}, self._stan_mcmc_args)

    def fit(self, model_name, model_param_names, data_input, fitter=None, init_values=None):
        compiled_stan_file = get_compiled_stan_model(model_name)

        #   passing callable from the model as seen in `initfun1()`
        #   https://pystan2.readthedocs.io/en/latest/api.html
        #   if None, use default as defined in class variable
        init_values = init_values or self.stan_init

        stan_mcmc_fit = compiled_stan_file.sampling(
            data=data_input,
            pars=model_param_names,
            iter=self._num_iter_per_chain,
            warmup=self._num_warmup_per_chain,
            chains=self.chains,
            n_jobs=self.cores,
            # fall back to default if not provided by model payload
            init=init_values,
            seed=self.seed,
            algorithm=self.algorithm,
            control=self.stan_mcmc_control,
            **self._stan_mcmc_args
        )

        # extract `lp__` in addition to defined model params
        model_param_names_with_lp = model_param_names[:] + ['lp__']

        stan_extract = stan_mcmc_fit.extract(
            pars=model_param_names_with_lp,
            permuted=False
        )

        # todo: move dimension cleaning function to the model directly
        for key, val in stan_extract.items():
            if len(val.shape) == 2:
                # here `order` is important to make samples flattened by chain
                stan_extract[key] = val.flatten(order='F')
            else:
                stan_extract[key] = val.reshape((-1, *val.shape[2:]), order='F')

        return stan_extract


class StanEstimatorVI(StanEstimator):
    """Stan Estimator for VI Sampling

    Parameters
    ----------
    stan_vi_args : dict
        Supplemental stan vi args to pass to PyStan.vb()


    """
    _is_mcmc_estimator = True

    def __init__(self, stan_vi_args=None, **kwargs):
        super().__init__(**kwargs)
        self.stan_vi_args = stan_vi_args

        # init internal variable
        self._stan_vi_args = copy(self.stan_vi_args)

        # set defaults if None
        self._set_computed_stan_vi_configs()

    def _set_computed_stan_vi_configs(self):
        default_stan_vi_args = {
            'iter': 10000,
            'grad_samples': 1,
            'elbo_samples': 100,
            'adapt_engaged': True,
            'tol_rel_obj': 0.01,
            'eval_elbo': 100,
            'adapt_iter': 50,
        }

        self._stan_vi_args = update_dict(default_stan_vi_args, self._stan_vi_args)

    @staticmethod
    def _vb_extract(vi_fit):
        """Re-arrange and extract posteriors from variational inference fit from stan

        Due to different structure of the output from fit from vb, we need this additional logic to
        extract posteriors.  The logic is based on
        https://gist.github.com/lwiklendt/9c7099288f85b59edc903a5aed2d2d64

        Parameters
        ----------
        vi_fit: dict
            dict exported from pystan.StanModel object by `vb` method

        Returns
        -------
        params: OrderedDict
            dict of arrays where each element represent arrays of samples (Index of Sample, Sample
            dimension 1, Sample dimension 2, ...)
        """
        param_specs = vi_fit['sampler_param_names']
        samples = vi_fit['sampler_params']
        n = len(samples[0])

        # first pass, calculate the shape
        param_shapes = OrderedDict()
        for param_spec in param_specs:
            splt = param_spec.split('[')
            name = splt[0]
            if len(splt) > 1:
                # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
                idxs = [int(i) for i in splt[1][:-1].split(',')]
            else:
                idxs = ()
            param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

        # create arrays
        params = OrderedDict([(name, np.nan * np.empty((n,) + tuple(shape))) for name, shape in param_shapes.items()])

        # second pass, set arrays
        for param_spec, param_samples in zip(param_specs, samples):
            splt = param_spec.split('[')
            name = splt[0]
            if len(splt) > 1:
                # -1 because pystan returns 1-based indexes for vb!
                idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]
            else:
                idxs = ()
            params[name][(...,) + tuple(idxs)] = param_samples

        return params

    def fit(self, model_name, model_param_names, data_input, fitter=None, init_values=None):
        compiled_stan_file = get_compiled_stan_model(model_name)

        #   passing callable from the model as seen in `initfun1()`
        #   https://pystan.readthedocs.io/en/latest/api.html
        #   if None, use default as defined in class variable
        init_values = init_values or self.stan_init

        stan_vi_fit = compiled_stan_file.vb(
            data=data_input,
            pars=model_param_names,
            init=init_values,
            seed=self.seed,
            algorithm=self.algorithm,
            output_samples=self.num_sample,
            **self._stan_vi_args
        )

        stan_extract = self._vb_extract(stan_vi_fit)  # `lp__` already automatically included for vb

        return stan_extract


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

    def fit(self, model_name, model_param_names, data_input, fitter=None, init_values=None):
        compiled_stan_file = get_compiled_stan_model(model_name)

        # passing callable from the model as seen in `initfun1()`
        init_values = init_values or self.stan_init

        # in case optimizing fails with given algorithm fallback to `Newton`
        try:
            stan_extract = compiled_stan_file.optimizing(
                data=data_input,
                init=init_values,
                seed=self.seed,
                algorithm=self.algorithm,
                **self._stan_map_args
            )
        except RuntimeError:
            self.algorithm = 'Newton'
            stan_extract = compiled_stan_file.optimizing(
                data=data_input,
                init=init_values,
                seed=self.seed,
                algorithm=self.algorithm,
                **self._stan_map_args
            )

        # make sure that model param names are a subset of stan extract keys
        invalid_model_param = set(model_param_names) - set(list(stan_extract.keys()))
        if invalid_model_param:
            raise EstimatorException("Stan model definition does not contain required parameters")

        # `stan.optimizing` automatically returns all defined parameters
        # filter out unecessary keys
        stan_extract = {param: stan_extract[param] for param in model_param_names}

        return stan_extract
