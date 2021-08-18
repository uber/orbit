from copy import deepcopy
import numpy as np
import pandas as pd
from enum import Enum

from ..exceptions import ForecasterException, AbstractMethodException
from ..utils.general import is_ordered_datetime
from ..template.model_template import ModelTemplate

COMMON_MODEL_CALLABLES = ['get_data_input_mapper', 'get_fitter', 'get_init_values', 'get_model_name',
                          'get_model_param_names', 'get_supported_estimator_types', 'predict',
                          'set_dynamic_attributes', 'set_init_values']


class Forecaster(object):
    def __init__(self,
                 model,
                 estimator_type,
                 response_col='y',
                 date_col='ds',
                 n_bootstrap_draws=-1,
                 prediction_percentiles=None,
                 **kwargs):
        """ Abstract class for providing template of how a forecaster works by containing a model under `ModelTemplate`

        Parameters
        ----------
        response_col : str
            Name of response variable column, default 'y'
        date_col : str
            Name of date variable column, default 'ds'
        n_bootstrap_draws : int
            Number of samples to bootstrap in order to generate the prediction interval. For full Bayesian and
            variational inference forecasters, samples are drawn directly from original posteriors. For point-estimated
            posteriors, it will be used to sample noise parameters.  When -1 or None supplied, full Bayesian and
            variational inference forecasters will assume number of draws equal the size of original samples while
            point-estimated posteriors will mute the draw and output prediction without interval.
        prediction_percentiles : list
            List of integers of prediction percentiles that should be returned on prediction. To avoid reporting any
            confident intervals, pass an empty list

        **kwargs:
            additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
        """

        # general fields
        if not isinstance(model, ModelTemplate):
            raise ForecasterException('Invalid class of model argument supplied.')
        self._model = model
        method_list = [
            attr for attr in dir(model)
            # only load public methods
            if (callable(getattr(model, attr))
                and not attr.startswith('__')
                and not attr.startswith('_')
                and attr not in COMMON_MODEL_CALLABLES)
        ]
        self.extra_methods = method_list
        self.response_col = response_col
        self.date_col = date_col
        self._validate_supported_estimator_type(estimator_type)
        self.estimator_type = estimator_type
        self.estimator = self.estimator_type(**kwargs)

        self.n_bootstrap_draws = n_bootstrap_draws
        # n_bootstrap_draws holds original input
        # _n_bootstrap_draws is validated input to handle default value

        if not self.n_bootstrap_draws:
            self._n_bootstrap_draws = -1
        else:
            self._n_bootstrap_draws = int(self.n_bootstrap_draws)

        self.prediction_percentiles = prediction_percentiles
        self._prediction_percentiles = None

        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = deepcopy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles))  # avoid duplicates
        self._prediction_percentiles.sort()

        # fields from fit and predict process
        # set by ._set_training_meta() in fit process
        self._training_meta = dict()
        self._training_data_input = dict()
        # fitted results after fit process
        self._posterior_samples = dict()
        self._point_posteriors = dict()
        self._training_metrics = None

        # set by ._set_prediction_meta() in predict process
        self._prediction_meta = dict()
        self.prediction_array = None

        # determine point approximation method of posteriors
        self._point_method = None

    # TODO: theoretically, we need to shrink estimator type to be consistent with the forecaster as well
    def _validate_supported_estimator_type(self, estimator_type):
        supported_estimator_types = self._model.get_supported_estimator_types()
        if estimator_type not in supported_estimator_types:
            msg_template = "Model class: {} is incompatible with Estimator: {}.  Estimator Support: {}"
            model_class = type(self._model)
            estimator_type = estimator_type
            raise ForecasterException(
                msg_template.format(model_class, estimator_type, str(supported_estimator_types))
            )

    def fit(self, df):
        """Core process for fitting a model within a forecaster"""
        estimator = self.estimator
        model_name = self._model.get_model_name()
        df = df.copy()

        # default set and validation of input data frame
        self._validate_training_df(df)
        # extract standard training metadata
        self._set_training_meta(df)
        # customize module
        self._model.set_dynamic_attributes(df=df, training_meta=self.get_training_meta())
        # based on the model and df, set training input
        self.set_training_data_input()
        # if model provide initial values, set it
        self._model.set_init_values()

        # estimator inputs
        data_input = self.get_training_data_input()
        init_values = self._model.get_init_values()
        model_param_names = self._model.get_model_param_names()

        # note that estimator will search for the .stan, .pyro model file based on the
        # estimator type and model_name provided
        _posterior_samples, training_metrics = estimator.fit(
            model_name=model_name,
            model_param_names=model_param_names,
            data_input=data_input,
            fitter=self._model.get_fitter(),
            init_values=init_values
        )

        self._posterior_samples = _posterior_samples
        self._training_metrics = training_metrics
        # load extra methods implemented in model layer after fitting
        # self.load_extra_methods()

    def _set_training_meta(self, df):
        """A default pre-processing and information gathering from training input dataframe"""
        training_meta = dict()
        response = df[self.response_col].values
        training_meta['response'] = response
        training_meta['date_array'] = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        training_meta['num_of_observations'] = len(response)
        training_meta['response_sd'] = np.nanstd(response)
        training_meta['training_start'] = df[self.date_col].iloc[0]
        training_meta['training_end'] = df[self.date_col].iloc[-1]
        # TODO: a little overhead here; there is room for further improvement
        training_meta['date_col'] = self.date_col
        training_meta['response_col'] = self.response_col
        self._training_meta = training_meta

    def get_training_meta(self):
        return deepcopy(self._training_meta)

    def set_training_data_input(self):
        """Collects data attributes into a dict for sampling/optimization api"""
        # refresh a clean dict
        data_input_mapper = self._model.get_data_input_mapper()
        if not data_input_mapper:
            raise ForecasterException('Empty or invalid data_input_mapper')

        # TODO: if we make a base class, this can be refactored
        # always get standard input from training
        training_meta = self.get_training_meta()
        training_data_input = {
            'RESPONSE': training_meta['response'],
            # response_sd not used in lgt/dlt
            'RESPONSE_SD': training_meta['response_sd'],
            'NUM_OF_OBS': training_meta['num_of_observations'],
            'WITH_MCMC': 1,
        }

        if isinstance(data_input_mapper, list):
            # if a list is provided, we assume an upper case in the mapper and reuse as the input value
            for key in data_input_mapper:
                key_lower = key.lower()
                input_value = getattr(self._model, key_lower, None)
                if input_value is None:
                    raise ForecasterException('{} is missing from data input'.format(key_lower))
                # stan accepts bool as int only
                if isinstance(input_value, bool):
                    input_value = int(input_value)
                training_data_input[key] = input_value
        elif issubclass(data_input_mapper, Enum):
            # isinstance(data_input_mapper, object):
            for key in data_input_mapper:
                # mapper keys in upper case; attributes is defined in lower case; need a cae casting in conversion
                key_lower = key.name.lower()
                input_value = getattr(self._model, key_lower, None)
                if input_value is None:
                    raise ForecasterException('{} is missing from data input'.format(key_lower))
                if isinstance(input_value, bool):
                    # stan accepts bool as int only
                    input_value = int(input_value)
                training_data_input[key.value] = input_value
        else:
            raise Exception('Invalid type: data_input_mapper needs to be either an Enum or list.')

        self._training_data_input = training_data_input

    def get_training_data_input(self):
        return self._training_data_input

    def _validate_training_df(self, df):
        df_columns = df.columns

        # validate date_col
        if self.date_col not in df_columns:
            raise ForecasterException("DataFrame does not contain `date_col`: {}".format(self.date_col))

        # validate ordering of time series
        date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        if not is_ordered_datetime(date_array):
            raise ForecasterException('Datetime index must be ordered and not repeat')

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ForecasterException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def is_fitted(self):
        # if either point posterior or posterior_samples are non-empty, claim it as fitted model (true),
        # else false.
        if bool(self._posterior_samples):
            return True
        for key in self._point_posteriors.keys():
            if bool(self._point_posteriors[key]):
                return True
        return False

    def predict(self, df, **kwargs):
        """Predict interface requires concrete implementation from child class"""
        raise AbstractMethodException("Abstract method.  Model should implement concrete .predict().")

    def _set_prediction_meta(self, df):
        """A default pre-processing and information gathering from prediction input dataframe"""
        # remove reference from original input
        df = df.copy()

        # get prediction df meta
        prediction_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'prediction_start': df[self.date_col].iloc[0],
            'prediction_end': df[self.date_col].iloc[-1],
        }

        if not is_ordered_datetime(prediction_meta['date_array']):
            raise ForecasterException('Datetime index must be ordered and not repeat')

        # TODO: validate that all regressor columns are present, if any

        if prediction_meta['prediction_start'] < self._training_meta['training_start']:
            raise ForecasterException('Prediction start must be after training start.')

        trained_len = self._training_meta['num_of_observations']

        # If we cannot find a match of prediction range, assume prediction starts right after train
        # end
        if prediction_meta['prediction_start'] > self._training_meta['training_end']:
            forecast_dates = set(prediction_meta['date_array'])
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = trained_len
        else:
            # compute how many steps to forecast
            forecast_dates = \
                set(prediction_meta['date_array']) - set(self._training_meta['date_array'])
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or - (
                len(set(self._training_meta['date_array']) - set(prediction_meta['date_array']))
            )
            # time index for prediction start
            start = pd.Index(
                self._training_meta['date_array']).get_loc(prediction_meta['prediction_start'])

        prediction_meta.update({
            'start': start,
            'n_forecast_steps': n_forecast_steps,
        })

        self._prediction_meta = prediction_meta

    def get_prediction_meta(self):
        return deepcopy(self._prediction_meta)

    def get_posterior_samples(self):
        return deepcopy(self._posterior_samples)

    def get_point_posteriors(self):
        return deepcopy(self._point_posteriors)

    def load_extra_methods(self):
        pass
