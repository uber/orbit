import pandas as pd
import numpy as np
import torch
from copy import copy, deepcopy

from ..constants import ets as constants
from ..constants.constants import PredictMethod
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP, StanEstimatorVI
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..initializer.ets import ETSInitializer
from .template import BaseTemplate, FullBayesianTemplate, AggregatedPosteriorTemplate, MAPTemplate
from ..utils.general import is_ordered_datetime


class BaseETS(BaseTemplate):
    """
    Parameters
    ----------
    seasonality : int
        Length of seasonality
    seasonality_sm_input : float
        float value between [0, 1], applicable only if `seasonality` > 1. A larger value puts
        more weight on the current seasonality.
        If None, the model will estimate this value.
    level_sm_input : float
        float value between [0.0001, 1]. A larger value puts more weight on the current level.
        If None, the model will estimate this value.

    Other Parameters
    ----------------
    **kwargs: additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
    """
    # data labels for sampler API (stan, pyro, numpyro etc.)
    _data_input_mapper = constants.DataInputMapper
    _model_name = 'ets'

    def __init__(self, seasonality=None, seasonality_sm_input=None, level_sm_input=None, **kwargs):
        super().__init__(**kwargs)  # create estimator in base class
        self.seasonality = seasonality

        # fixed smoothing parameters config
        self.seasonality_sm_input = seasonality_sm_input
        self.level_sm_input = level_sm_input

        # indicator of using mcmc
        self._with_mcmc = 0

        # set private var to arg value
        # if None set default in _set_default_base_args()
        self._seasonality = self.seasonality
        self._seasonality_sm_input = self.seasonality_sm_input
        self._level_sm_input = self.level_sm_input

        self._training_df_meta = None
        self._model_data_input = dict()

        # depends on seasonality length
        self._init_values = None

        # set static data attributes
        self._set_static_data_attributes()

        # set model param names
        # this only depends on static attributes, but should these params depend
        # on dynamic data (e.g actual data matrix, number of responses, etc) this should be
        # called after fit instead
        self._set_model_param_names()

        # init dynamic data attributes
        # the following are set by `_set_dynamic_data_attributes()` and generally set during fit()
        # from input df
        # response data
        self._response = None
        self._num_of_observations = None
        self._repsonse_sd = None

        # init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }

    def _set_default_base_args(self):
        """Cast data to the proper type mostly to match Stan required static data types
        """
        if self.seasonality_sm_input is None:
            self._seasonality_sm_input = -1
        if self.level_sm_input is None:
            self._level_sm_input = -1
        elif self.level_sm_input < 0.0001 or self.level_sm_input > 1:
            raise IllegalArgument('only values between [0.0001, 1] are supported for level_sm_input '
                                  'to build a model with meaningful trend.')
        if self.seasonality is None:
            self._seasonality = -1

    def _set_with_mcmc(self):
        """Include extra indicator to indicate whether the object is using mcmc type of estimator
        """
        estimator_type = self.estimator_type
        # set `_with_mcmc` attribute based on estimator type
        # if no attribute for _is_mcmc_estimator, default to False
        if getattr(estimator_type, '_is_mcmc_estimator', False):
            self._with_mcmc = 1

    def _set_init_values(self):
        """Set init as a callable (for Stan ONLY)
        See: https://pystan.readthedocs.io/en/latest/api.htm
        """
        # init_values_partial = partial(init_values_callable, seasonality=seasonality)
        # partialfunc does not work when passed to PyStan because PyStan uses
        # inspect.getargspec(func) which seems to raise an exception with keyword-only args
        # caused by using partialfunc
        # lambda does not work in serialization in pickle
        # callable object as an alternative workaround
        if self._seasonality > 1:
            init_values_callable = ETSInitializer(
                self._seasonality
            )
            self._init_values = init_values_callable

    def _set_static_data_attributes(self):
        """Set data input based on args or computed from args at object instantiation"""
        self._set_default_base_args()
        self._set_with_mcmc()
        self._set_init_values()

    def _set_training_df_meta(self, df):
        # Date Metadata
        # TODO: use from constants for dict key
        self._training_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'training_start': df[self.date_col].iloc[0],
            'training_end': df[self.date_col].iloc[-1]
        }

    def _validate_training_df(self, df):
        df_columns = df.columns

        # validate date_col
        if self.date_col not in df_columns:
            raise ModelException("DataFrame does not contain `date_col`: {}".format(self.date_col))

        # validate ordering of time series
        date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        if not is_ordered_datetime(date_array):
            raise ModelException('Datetime index must be ordered and not repeat')

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ModelException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def _set_dynamic_data_attributes(self, df):
        """Set required input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()

        self._validate_training_df(df)
        self._set_training_df_meta(df)

        # a few of the following are related with training data.
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)
        self._response_sd = np.std(self._response)
        self._set_init_values()

    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]

        # append seasonality param names
        if self._seasonality > 1:
            self._model_param_names += [param.value for param in constants.SeasonalitySamplingParameters]

    def _get_model_param_names(self):
        return self._model_param_names

    def _set_model_data_input(self):
        """Collects data attributes into a dict for sampling/optimization api"""
        data_inputs = dict()

        for key in self._data_input_mapper:
            # mapper keys in upper case; inputs in lower case
            key_lower = key.name.lower()
            input_value = getattr(self, key_lower, None)
            if input_value is None:
                raise ModelException('{} is missing from data input'.format(key_lower))
            if isinstance(input_value, bool):
                # stan accepts bool as int only
                input_value = int(input_value)
            data_inputs[key.value] = input_value

        self._model_data_input = data_inputs

    def _get_model_data_input(self):
        return self._model_data_input

    def _get_init_values(self):
        return self._init_values

    def _predict(self, posterior_estimates, df, include_error=False, decompose=False):
        """Vectorized version of prediction math"""

        ################################################################
        # Model Attributes
        ################################################################

        model = deepcopy(posterior_estimates)
        for k, v in model.items():
            model[k] = torch.from_numpy(v)

        # We can pull any arbitrary value from teh dictionary because we hold the
        # safe assumption: the length of the first dimension is always the number of samples
        # thus can be safely used to determine `num_sample`. If predict_method is anything
        # other than full, the value here should be 1
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        # seasonality components
        seasonality_levels = model.get(
            constants.SeasonalitySamplingParameters.SEASONALITY_LEVELS.value)
        seasonality_smoothing_factor = model.get(
            constants.SeasonalitySamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend componentsð
        level_smoothing_factor = model.get(
            constants.BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        local_trend_levels = model.get(constants.BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        residual_sigma = model.get(constants.BaseSamplingParameters.RESIDUAL_SIGMA.value)

        ################################################################
        # Prediction Attributes
        ################################################################

        # get training df meta
        training_df_meta = self._training_df_meta
        # remove reference from original input
        df = df.copy()

        # get prediction df meta
        prediction_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'prediction_start': df[self.date_col].iloc[0],
            'prediction_end': df[self.date_col].iloc[-1]
        }

        if not is_ordered_datetime(prediction_df_meta['date_array']):
            raise IllegalArgument('Datetime index must be ordered and not repeat')

        # TODO: validate that all regressor columns are present, if any

        if prediction_df_meta['prediction_start'] < training_df_meta['training_start']:
            raise PredictionException('Prediction start must be after training start.')

        trained_len = training_df_meta['df_length']

        # If we cannot find a match of prediction range, assume prediction starts right after train
        # end
        if prediction_df_meta['prediction_start'] > training_df_meta['training_end']:
            forecast_dates = set(prediction_df_meta['date_array'])
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = trained_len
        else:
            # compute how many steps to forecast
            forecast_dates = \
                set(prediction_df_meta['date_array']) - set(training_df_meta['date_array'])
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or \
                -(len(set(training_df_meta['date_array']) - set(prediction_df_meta['date_array'])))
            # time index for prediction start
            start = pd.Index(
                training_df_meta['date_array']).get_loc(prediction_df_meta['prediction_start'])

        full_len = trained_len + n_forecast_steps

        ################################################################
        # Seasonality Component
        ################################################################

        # calculate seasonality component
        if self._seasonality > 1:
            if full_len <= seasonality_levels.shape[1]:
                seasonal_component = seasonality_levels[:, :full_len]
            else:
                seasonality_forecast_length = full_len - seasonality_levels.shape[1]
                seasonality_forecast_matrix \
                    = torch.zeros((num_sample, seasonality_forecast_length), dtype=torch.double)
                seasonal_component = torch.cat(
                    (seasonality_levels, seasonality_forecast_matrix), dim=1)
        else:
            seasonal_component = torch.zeros((num_sample, full_len), dtype=torch.double)

        ################################################################
        # Trend Component
        ################################################################

        # calculate level component.
        # However, if predicted end of period > training period, update with out-of-samples forecast
        if full_len <= trained_len:
            trend_component = local_trend_levels[:, :full_len]
            # in-sample error are iids
            if include_error:
                error_value = np.random.normal(
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=trend_component.shape
                )

                error_value = torch.from_numpy(error_value).double()
                trend_component += error_value
        else:
            trend_component = local_trend_levels
            # in-sample error are iids
            if include_error:
                error_value = np.random.normal(
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=trend_component.shape
                )

                error_value = torch.from_numpy(error_value).double()
                trend_component += error_value

            trend_forecast_matrix = torch.zeros((num_sample, n_forecast_steps), dtype=torch.double)
            trend_component = torch.cat((trend_component, trend_forecast_matrix), dim=1)

            last_local_trend_level = local_trend_levels[:, -1]

            for idx in range(trained_len, full_len):
                trend_component[:, idx] = last_local_trend_level

                if include_error:
                    error_value = np.random.normal(
                        scale=residual_sigma,
                        size=num_sample
                    )
                    error_value = torch.from_numpy(error_value).double()
                    trend_component[:, idx] += error_value

                new_local_trend_level = \
                    level_smoothing_factor * trend_component[:, idx] \
                    + (1 - level_smoothing_factor) * last_local_trend_level

                if self._seasonality > 1 and idx + self._seasonality < full_len:
                    seasonal_component[:, idx + self._seasonality] = \
                        seasonality_smoothing_factor.flatten() \
                        * (trend_component[:, idx] + seasonal_component[:, idx] -
                           new_local_trend_level) \
                        + (1 - seasonality_smoothing_factor.flatten()) * seasonal_component[:, idx]

                last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = trend_component[:, start:]
        seasonal_component = seasonal_component[:, start:]

        # sum components
        pred_array = trend_component + seasonal_component

        pred_array = pred_array.numpy()
        trend_component = trend_component.numpy()
        seasonal_component = seasonal_component.numpy()

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend_component,
                'seasonality': seasonal_component
            }

            return decomp_dict

        return {'prediction': pred_array}

    def _set_aggregate_posteriors(self):
        posterior_samples = self._posterior_samples

        mean_posteriors = {}
        median_posteriors = {}

        # for each model param, aggregate using `method`
        for param_name in self._model_param_names:
            param_ndarray = posterior_samples[param_name]

            mean_posteriors.update(
                {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
            )

            median_posteriors.update(
                {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
            )

        self._aggregate_posteriors[PredictMethod.MEAN.value] = mean_posteriors
        self._aggregate_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        estimator = self.estimator
        model_name = self._model_name

        self._set_dynamic_data_attributes(df)
        self._set_model_data_input()

        # estimator inputs
        data_input = self._get_model_data_input()
        init_values = self._get_init_values()
        model_param_names = self._get_model_param_names()

        model_extract = estimator.fit(
            model_name=model_name,
            model_param_names=model_param_names,
            data_input=data_input,
            init_values=init_values
        )

        self._posterior_samples = model_extract


class ETSMAP(BaseETS, MAPTemplate):
    """Concrete ETS model for MAP (Maximum a Posteriori) prediction

    Similar to `ETSAggregated` but prediction is based on Maximum a Posteriori (aka Mode)
    of the posterior.

    This model only supports MAP estimating `estimator_type`s

    """
    _supported_estimator_types = [StanEstimatorMAP]

    def __init__(self, estimator_type=StanEstimatorMAP, **kwargs):
        super().__init__(estimator_type=estimator_type, **kwargs)

    def fit(self, df):
        super().fit(df)
        self._set_map_posterior()

    def predict(self, df, decompose=False, **kwargs):
        return self._map_predict(df, self._predict, decompose=decompose)


class ETSFull(BaseETS, FullBayesianTemplate):
    """Concrete ETS model for full Bayesian prediction"""
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, df, decompose=False, **kwargs):
        return self._full_bayes_predict(df, self._predict, decompose=decompose)


class ETSAggregated(BaseETS, AggregatedPosteriorTemplate):
    """Concrete ETS model for aggregated posterior prediction"""
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, df):
        super().fit(df)
        self._set_aggregate_posteriors()

    def predict(self, df, decompose=False, **kwargs):
        return self._aggregate_predict(df, self._predict, decompose=decompose)


