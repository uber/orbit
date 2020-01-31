from abc import ABCMeta, abstractmethod
import copy
from copy import deepcopy
import inspect
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from orbit.models import get_compiled_stan_model
from orbit.exceptions import (
    IllegalArgument
)
from orbit.pyro.wrapper import pyro_map, pyro_svi
from orbit.utils.constants import (
    PredictMethod,
    SampleMethod,
    StanInputMapper
)

from orbit.utils.utils import vb_extract, is_ordered_datetime


class Estimator(object):
    """The abstract base mix-in class for stan Bayesian time series models.

    This module contains the implementation for the common interfaces and methods
    that are to be inherited by all concrete model implementations.

    Parameters
    ----------
    response_col : str
        Name of the response column
    date_col : st
        Name of the date index column
    num_warmup : int
        Number of warm up iterations for MCMC sampler
    num_sample : int
        Number of samples to extract for MCMC sampler
    chains : int
        Number of MCMC chains
    cores : int
        Number of cores. If cores is set higher than the CPU count of the system
        the lower of the two numbers will be used.
    stan_control : dict
        Dictionary contains additional settings to call pystan
    seed : int
        Used to set the seed of the random init
    predict_method : {'mean', 'median', 'full', 'map'}
        Determines which stan interface to use during fit (e.g. sampling / optimizing)
        and how to aggregate and return predicted values.
    n_boostrap_draws : int, default -1
        Number of bootstrap samples to draw from `posterior_samples`
    prediction_percentiles : list
        Prediction percentiles which should be returned in addition to predicted values.
        This is only valid when `predict_method` is 'full'
    verbose : bool
        Print verbose.

    Attributes
    ----------
    df : pd.DataFrame
        Training DataFrame
    training_df_meta : dict
        Training DataFrame meta data
    raw_stan_extract : dict
        Raw format for the posterior samples extracted from stan sampler. Key
        is a model param name and value is either a 1d or 2d array, depending
        on if the sampled param is a scalar or vector.
    posterior_samples : list of dict
        Inverted data structure of `raw_stan_extract` so that each element is
        a dict containing a single posterior sample.
    aggregated_posteriors : dict
        {'mean', 'median'} aggregation of `posterior_samples`
    stan_model_name : str
        stan model name set by the concrete child class. This is used to find the
        corresponding `.stan` model file.

    Notes
    -----
        The child class must implement the following methods

            - _set_model_param_names()
            - _set_dynamic_inputs()
            - _predict_once()
            - _validate_params() [optional]
            - _set_computed_params() [optional]
            - _plot() [optional]

    """
    __metaclass__ = ABCMeta

    def __init__(
            self, response_col='y', date_col='ds',
            num_warmup=900, num_sample=100, chains=4, cores=8, stan_control=None, seed=8888,
            predict_method="full", sample_method="mcmc", n_bootstrap_draws=-1,
            prediction_percentiles=[5, 95], algorithm=None,
            # vi additional parameters
            max_iter=10000, grad_samples=1, elbo_samples=100, adapt_engaged=True,
            tol_rel_obj=0.01, eval_elbo=100, adapt_iter=50,
            inference_engine='stan', verbose=False, **kwargs
    ):

        # TODO: mutable defaults are dangerous. Use sentinel value
        #   https://docs.python-guide.org/writing/gotchas/

        # get all init args and values and set
        local_params = {k: v for (k, v) in locals().items() if k not in ['kwargs', 'self']}
        kw_params = locals()['kwargs']

        # kwargs is strictly used to pass parent param
        # and so we should never receive params that are not explicitly defined in __init__
        if len(kw_params.keys()) > 0:
            raise IllegalArgument('Received an undefined init param: {}'.format(kw_params))

        self.set_params(**local_params)

        # mcmc config derived from __init__
        self._derive_sampler_config()

        # training DataFrame
        self.df = None
        self.training_df_meta = {}

        # posterior samples as a result of stan extract
        # these are computed in `self._set_posterior_samples`
        self.posterior_samples = []
        self.raw_stan_extract = None

        # aggregated posteriors
        # although this is only necessary if predict_method != 'full'
        # we save these at time of extracting stan samples for computational efficiency
        self.aggregated_posteriors = {
            PredictMethod.MEAN.value: None,
            PredictMethod.MEDIAN.value: None,
            PredictMethod.MAP.value: None
        }

        # posterior state for a single prediction call
        self._posterior_state = {}

        # stan model inputs
        self.stan_inputs = {}

        # stan model name
        self.stan_model_name = ''

        # stan model parameters names
        self.model_param_names = []

        self.stan_init = 'random'

        # set computed params
        self._set_computed_params()

    def get_params(self):
        """Get all class attributes and values"""
        out = dict()

        for key in self._get_param_names():
            if not hasattr(self, key):
                continue
            out[key] = getattr(self, key)

        return out

    @classmethod
    def _get_param_names(cls):
        """Get all class attribute names

        This method gets class attribute names for `cls` and child classes.
        """

        # Return a tuple of class cls’s base classes
        # including cls, in method resolution order (mro)
        class_hierarchy = inspect.getmro(cls)

        # inspect.signature() of cls and all parents
        # excluding `object` base class
        init_signatures = \
            [inspect.signature(c.__init__) for c in class_hierarchy if c is not object]

        # unpacked into flat list of params
        all_params = []
        for sig in init_signatures:
            params = [p for p in sig.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD
                      and p.kind != p.VAR_POSITIONAL]
            all_params.extend(params)

        param_names = [p.name for p in all_params]

        return param_names

    def set_params(self, **params):
        """Sets all class attributes."""
        # TODO: check for valid parameter names
        for key, value in params.items():
            setattr(self, key, value)

    def save(self, path='./uts_model.pkl'):
        """Serializes the entire `orbit` class to `path`"""
        # todo: this method needs to be fixed
        try:
            self.posterior_samples
            if not self.posterior_samples:
                raise Exception("Stan model hasn't been trained yet. Nothing to save.")
        except:
            # TODO: implement real exceptions
            raise ValueError(
                "Stan model hasn't been trained yet. Nothing to save. Please run fit() first.",
            )
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Loads the serialized model from `path`"""
        with open(path, 'rb') as f:
            uts_object = pickle.load(f)

        return uts_object

    def _derive_sampler_config(self):
        """Sets sampler configs based on init class attributes"""
        # make sure cores can only be as large as the device support
        self.cores = min(self.cores, multiprocessing.cpu_count())
        self.num_warmup_per_chain = int(self.num_warmup/self.chains)
        self.num_sample_per_chain = int(self.num_sample/self.chains)
        self.num_iter_per_chain = self.num_warmup_per_chain + self.num_sample_per_chain
        self.total_iter = self.num_iter_per_chain * self.chains

        if self.verbose:
            print(
                "Using {} chains, {} cores, {} warmup and {} samples per chain for sampling.". \
                    format(self.chains, self.cores, self.num_warmup_per_chain,
                           self.num_sample_per_chain)
            )

    def fit(self, df):
        """Estimates the model posterior values

        This method will utilize the following implementations from the child class

            - _set_model_param_names()
            - _set_dynamic_inputs()
            - _validate_params() [optional]

        Args
        ----
        df : pd.DataFrame
            The DataFrame on which the model is fit. Both `date_col` and `response_col`
            are required columns in the DataFrame. If the child class models allows regressors,
            DataFrame must contain all regressor columns.

        Notes
        -----
        The general orchestration of `fit()` is the following:

        #. Prior to `fit()` all static input stan parameters are already set
        #. `_set_dynamic_inputs()` sets stan input parameters that are based on `df`
        #. `_validate_params()` checks consistency of input between static and dynamic inputs
        #. stan input parameters are collected and set into a stan input dictionary
        #. `_set_model_param_names()` sets the parameter names for which to gather posterior values
        #. `predict_method` class attribute determines which stan method to execute
        #. Executes stan sampler or estimator and stores posterior values to `posterior_samples`
        """

        # Date Metadata
        # TODO: use from constants for dict key
        self.training_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'training_start': df[self.date_col].iloc[0],
            'training_end': df[self.date_col].iloc[-1]
        }

        if not is_ordered_datetime(self.training_df_meta['date_array']):
            raise IllegalArgument('Datetime index must be ordered and not repeat')

        self.df = df.copy()

        self._set_dynamic_inputs()
        self._validate_params()

        self._convert_to_stan_inputs()

        # stan model parameters
        self._set_model_param_names()

        if self.inference_engine == 'stan':

            compiled_stan_file = get_compiled_stan_model(self.stan_model_name)

            if self.predict_method == PredictMethod.MAP.value:
                try:
                    stan_extract = compiled_stan_file.optimizing(
                        data=self.stan_inputs,
                        init=self.stan_init,
                        seed=self.seed,
                        algorithm=self.algorithm
                    )
                except RuntimeError:
                    self.algorithm = 'Newton'
                    stan_extract = compiled_stan_file.optimizing(
                        data=self.stan_inputs,
                        init=self.stan_init,
                        seed=self.seed,
                        algorithm=self.algorithm
                    )
                self._set_map_posterior(stan_extract=stan_extract)
            elif self.sample_method == SampleMethod.VARIATIONAL_INFERENCE.value:
                stan_extract = vb_extract(compiled_stan_file.vb(
                    data=self.stan_inputs,
                    pars=self.model_param_names,
                    iter=self.max_iter,
                    output_samples=self.num_sample,
                    init=self.stan_init,
                    seed=self.seed,
                    algorithm=self.algorithm,
                    grad_samples=self.grad_samples,
                    elbo_samples=self.elbo_samples,
                    adapt_engaged=self.adapt_engaged,
                    tol_rel_obj=self.tol_rel_obj,
                    eval_elbo=self.eval_elbo,
                    adapt_iter=self.adapt_iter
                ))
                # set posterior samples instance var
                self._set_aggregate_posteriors(stan_extract=stan_extract)
            elif self.sample_method == SampleMethod.MARKOV_CHAIN_MONTE_CARLO.value:
                stan_extract = compiled_stan_file.sampling(
                    data=self.stan_inputs,
                    pars=self.model_param_names,
                    iter=self.num_iter_per_chain,
                    warmup=self.num_warmup_per_chain,
                    chains=self.chains,
                    n_jobs=self.cores,
                    init=self.stan_init,
                    seed=self.seed,
                    algorithm=self.algorithm,
                    control=self.stan_control
                ).extract(permuted=True)
                # set posterior samples instance var
                self._set_aggregate_posteriors(stan_extract=stan_extract)
            else:
                raise NotImplementedError('Invalid sampling/predict method supplied.')

            self.posterior_samples = stan_extract

        elif self.inference_engine == 'pyro':
            if self.predict_method == 'map':
                pyro_extract = pyro_map(
                    model_name="orbit.pyro.lgt.LGTModel",
                    data=self.stan_inputs,
                    seed=self.seed,
                )
                self._set_map_posterior(stan_extract=pyro_extract)

            elif self.predict_method in ['svi', 'mean', 'median']:
                pyro_extract = pyro_svi(
                    model_name="orbit.pyro.lgt.LGTModel",
                    data=self.stan_inputs,
                    seed=self.seed,
                    num_samples=self.num_sample,
                )
                self._set_aggregate_posteriors(stan_extract=pyro_extract)

            else:
                raise ValueError(
                    'Pyro inferece does not support prediction method: "{}"'.format(
                        self.predict_method))

            self.posterior_samples = pyro_extract

        else:
            raise ValueError('Unknown inference engine: "{}"'.format(self.inference_engine))

    @abstractmethod
    def plot(self):
        """Plots the prediction results"""
        raise NotImplementedError('No Plot Method is Implemented For this Model')

    @abstractmethod
    def _predict(self, df=None, decompose=False):
        """Prediction for each set of parameters

        This defines a prediction with a set of parameters.  It should be called within `predict()`
        Child class should inherit this to define its own prediction process.

        Args
        ----
        df : pd.DataFrame
            The DataFrame on which prediction occurs. `date_col` is a required column
            in `df` for which the return object will align with. If the child class
            models allows regressors, `df` must contain all regressor columns.
        decompose : bool
            If True, prediction return includes columns for prediction value decomposition
            in addition to the prediction value.


        Returns
        -------
        dict:
            dict of predictions and/or components

        Notes
        -----
        The core prediction math to be implemented in the child class.
        """

        raise NotImplementedError('_predict must be implemented in the child class')

    def _bootstrap(self, n):
        """Draw `n` number of bootstrap samples from the posterior_samples.

        Args
        ----
        n : int
            The number of bootstrap samples to draw

        Raises
        ------
        IllegalArgument
            If `predict_method` is not 'mcmc'

        """
        if self.predict_method != 'full':
            raise IllegalArgument(
                'Error: Bootstrap is only supported for full Bayesian predict method'
            )  # pragma: no cover

        if n < 2:
            raise IllegalArgument("Error: The number of bootstrap draws must be at least 2")

        sample_idx = np.random.choice(
            range(self.num_sample),
            size=n,
            replace=True
        )

        bootstrap_samples_dict = {}
        for k, v in self.posterior_samples.items():
            bootstrap_samples_dict[k] = v[sample_idx]

        return bootstrap_samples_dict

    def _set_aggregate_posteriors(self, stan_extract):
        """Aggregates the raw stan extract to a point estimate using {'mean', 'median'}

        Args
        ----
        stan_extract : dict
            A dict of numpy ndarrays, in which each key is a sampled model param name
            as determined by `_set_model_param_names()`

        """

        mean_posteriors = {}
        median_posteriors = {}

        # for each model param, aggregate using `method`
        for param_name in self.model_param_names:
            param_ndarray = stan_extract[param_name]

            mean_posteriors.update(
                {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
            )

            median_posteriors.update(
                {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
            )

        self.aggregated_posteriors[PredictMethod.MEAN.value] = mean_posteriors
        self.aggregated_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

    @staticmethod
    def _aggregate_full_predictions(predictions_array, percentiles=[]):
        """Aggregates the mcmc prediction to a point estimate

        Args
        ----
        predictions_array : np.ndarray
            A 2d numpy array of shape (`num_samples`, prediction df length)
        percentiles : list
            The percentiles at which to aggregate the predictions

        Returns
        -------
        pd.DataFrame
            The aggregated across mcmc samples with columns for `mean`, `50` aka median
            and all other percentiles specified in `percentiles`.

        """

        # todo: default value for `percentiles` None instead of []
        # MUST copy, or else instance var persists in memory
        percentiles = copy.copy(percentiles)

        percentiles += [50]  # always find median
        percentiles.sort()

        # mean_prediction = np.mean(predictions_array, axis=0)
        percentiles_prediction = np.percentile(predictions_array, percentiles, axis=0)

        aggregate_df = pd.DataFrame(percentiles_prediction.T, columns=percentiles)
        # aggregate_df['mean'] = mean_prediction

        return aggregate_df

    def predict(self, df=None, decompose=False):
        """Predicts the response column given `df`

        This method will utilize the following implementations from the child class

            - _predict_once()

            A valid prediction must contain consecutive time index and the start date
            must occur between the training data start and end date.

        Args
        ----
        df : pd.DataFrame
            The DataFrame on which prediction occurs. `date_col` is a required column
            in `df` for which the return object will align with. If the child class
            models allows regressors, `df` must contain all regressor columns.
        decompose : bool
            If True, prediction return includes columns for prediction value decomposition
            in addition to the prediction value.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the prediction values. DataFrame will contain `date_col`
            identical to input `df`. Prediction values will be set by the class
            attribute `predict_method`.

        """

        if self.predict_method == PredictMethod.FULL_SAMPLING.value:

            # decompose not supported for MCMC
            if decompose:
                raise IllegalArgument('Decomposition is not supported for MCMC prediction')

            # if bootstrap draws, replace posterior samples with bootstrap
            posterior_samples = self._bootstrap(self.n_bootstrap_draws) \
                if self.n_bootstrap_draws > 1 \
                else self.posterior_samples

            self._posterior_state = posterior_samples

            predictions_array \
                = self._predict(df=df, include_error=True)['prediction']

            aggregated_df = self._aggregate_full_predictions(
                predictions_array,
                percentiles=self.prediction_percentiles
            )

            aggregated_df = self._prepend_date_column(aggregated_df, df)

            return aggregated_df

        # Predict Method: Other Aggregation
        else:
            self._posterior_state = self.aggregated_posteriors.get(self.predict_method)

            # prediction
            predicted_dict = self._predict(
                df=df, include_error=False, decompose=decompose
            )

            # must flatten to convert to DataFrame
            for k, v in predicted_dict.items():
                predicted_dict[k] = v.flatten()

            predicted_df = pd.DataFrame(predicted_dict)

            predicted_df = self._prepend_date_column(predicted_df, df)

            return predicted_df

    def _prepend_date_column(self, predicted_df, input_df):
        """Prepends date column from `input_df` to `predicted_df`"""

        other_cols = list(predicted_df.columns)

        # add date column
        predicted_df[self.date_col] = input_df[self.date_col].reset_index(drop=True)

        # re-order columns so date is first
        col_order = [self.date_col] + other_cols
        predicted_df = predicted_df[col_order]

        return predicted_df

    @abstractmethod
    def _set_computed_params(self):
        """Sets attributes that are computed based on __init__ params

        This is an optional method that must be implemented in the child class, if any
        computed params exist.

        """
        pass

    @abstractmethod
    def _set_dynamic_inputs(self):
        """Sets attributes that are dependent on training `df`

        This method must be implemented in the child class.

        """
        raise NotImplementedError('_validate_params must be implemented in the child class')

    def _convert_to_stan_inputs(self):
        """Collects stan attributes into a dict for `StanModel.sampling`"""
        # todo: this should probably not be in the base class
        #   and constants StanInputMapper should be model specific
        stan_input_set = set([each.name for each in StanInputMapper])
        stan_inputs = {}
        for key, value in self.__dict__.items():
            key = key.upper()
            if key not in stan_input_set:
                continue
                # TODO: constants for attributes not meant to be in stan input explicit exceptions
                # raise Exception('')
            stan_inputs[StanInputMapper[key].value] = value
        self.stan_inputs = stan_inputs

    @abstractmethod
    def _set_model_param_names(self):
        """Sets the stan model parameter names for which we want to return posterior samples

        This method must be implemented in the child class, and will set `model_param_names`

        """
        raise NotImplementedError('_validate_params must be implemented in the child class')

    def _set_map_posterior(self, stan_extract):
        """Sets `posterior_samples` after fit for 'MAP' predict method.

        Args
        ----
        stan_extract : dict
            The raw data extract from `StanModel.optimizing()`.

        """

        map_posterior = {}
        for param_name in self.model_param_names:
            param_array = stan_extract[param_name]
            # add dimension so it works with vector math in `_predict`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update({param_name: param_array})

        self.aggregated_posteriors[PredictMethod.MAP.value] = map_posterior

    @abstractmethod
    def _validate_params(self):
        """Validates static and dynamic input parameters

        This method must be implemented in the child class.

        """
        raise NotImplementedError('_validate_params must be implemented in the child class')
