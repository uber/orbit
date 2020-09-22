import pandas as pd
import numpy as np
from scipy.stats import nct
import torch
from copy import copy, deepcopy

from ..constants import lgt as constants
from ..constants.constants import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA,
    COEFFICIENT_DF_COLS,
    PredictMethod
)
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from ..estimators.pyro_estimator import PyroEstimatorVI, PyroEstimatorMAP
from ..exceptions import IllegalArgument, ModelException, PredictionException
from .base_model import BaseModel
from ..utils.general import is_ordered_datetime


class BaseLGT(BaseModel):
    """Base LGT model object with shared functionality for Full, Aggregated, and MAP methods

    Parameters
    ----------
    response_col : str
        Name of response variable column, default 'y'
    date_col : str
        Name of date variable column, default 'ds'
    regressor_col : list
        Names of regressor columns, if any
    seasonality : int
        Length of seasonality
    regressor_sign :  list
        list with values { '+', '=' }. '+' indicates regressor coefficient estimates are
        constrained to [0, inf). '=' indicates regressor coefficient estimates
        can be any value between (-inf, inf). The length of `regressor_sign` must be
        the same length as `regressor_col`. If None, all elements of list will be set
        to '='.
    regressor_beta_prior : list
        list of prior float values for regressor coefficient betas. The length of `regressor_beta_prior`
        must be the same length as `regressor_col`. If None, use non-informative priors.
    regressor_sigma_prior : list
        list of prior float values for regressor coefficient sigmas. The length of `regressor_sigma_prior`
        must be the same length as `regressor_col`. If None, use non-informative priors.
    regression_penalty : { 'fixed_ridge', 'lasso', 'auto_ridge' }
        regression penalty method
    lasso_scale : float
        float value between [0, 1], applicable only if `regression_penalty` == 'lasso'
    auto_ridge_scale : float
        float value between [0, 1], applicable only if `regression_penalty` == 'auto_ridge'
    seasonality_sm_input : float
        float value between [0, 1], applicable only if `seasonality` > 1. A larger value puts
        more weight on the current seasonality.
        If None, the model will estimate this value.
    slope_sm_input : float
        float value between [0, 1]. A larger value puts more weight on the current slope.
        If None, the model will estimate this value.
    level_sm_input : float
        float value between [0, 1]. A larger value puts more weight on the current level.
        If None, the model will estimate this value.
    kwargs
        To specify `estimator_type` or additional args for the specified `estimator_type`

    """
    _data_input_mapper = constants.DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'lgt'
    _supported_estimator_types = None  # set for each model

    def __init__(self, response_col='y', date_col='ds', regressor_col=None,
                 seasonality=None,
                 regressor_sign=None, regressor_beta_prior=None, regressor_sigma_prior=None,
                 regression_penalty='fixed_ridge', lasso_scale=0.5, auto_ridge_scale=0.5,
                 seasonality_sm_input=None, slope_sm_input=None, level_sm_input=None,
                 **kwargs):
        super().__init__(**kwargs)  # create estimator in base class
        self.response_col = response_col
        self.date_col = date_col
        self.regressor_col = regressor_col
        self.seasonality = seasonality
        self.regressor_sign = regressor_sign
        self.regressor_beta_prior = regressor_beta_prior
        self.regressor_sigma_prior = regressor_sigma_prior
        self.regression_penalty = regression_penalty
        self.lasso_scale = lasso_scale
        self.auto_ridge_scale = auto_ridge_scale
        self.seasonality_sm_input = seasonality_sm_input
        self.slope_sm_input = slope_sm_input
        self.level_sm_input = level_sm_input

        # set private var to arg value
        # if None set default in _set_default_base_args()
        self._seasonality = self.seasonality
        self._seasonality_sm_input = self.seasonality_sm_input
        self._slope_sm_input = self.slope_sm_input
        self._level_sm_input = self.level_sm_input
        self._regressor_sign = self.regressor_sign
        self._regressor_beta_prior = self.regressor_beta_prior
        self._regressor_sigma_prior = self.regressor_sigma_prior

        self._model_param_names = list()
        self._training_df_meta = None
        self._model_data_input = dict()

        # init static data attributes
        # the following are set by `_set_static_data_attributes()`
        self._regression_penalty = None
        self._with_mcmc = 0
        # todo: should this be based on number of obs?
        self._min_nu = 5.
        self._max_nu = 40.
        # positive regressors
        self._num_of_positive_regressors = 0
        self._positive_regressor_col = list()
        self._positive_regressor_beta_prior = list()
        self._positive_regressor_sigma_prior = list()
        # regular regressors
        self._num_of_regular_regressors = 0
        self._regular_regressor_col = list()
        self._regular_regressor_beta_prior = list()
        self._regular_regressor_sigma_prior = list()
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
        self._cauchy_sd = None
        # regression data
        self._positive_regressor_matrix = None
        self._regular_regressor_matrix = None

        # init posterior samples
        # `_posterior_samples` is set by `fit()`
        self._posterior_samples = dict()

        # init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }
        
    def _set_default_base_args(self):
        """Set default attributes for None

        Stan requires static data types so data must be cast to the correct type
        """
        if self.seasonality_sm_input is None:
            self._seasonality_sm_input = -1
        if self.slope_sm_input is None:
            self._slope_sm_input = -1
        if self.level_sm_input is None:
            self._level_sm_input = -1
        if self.seasonality is None:
            self._seasonality = -1

        ##############################
        # if no regressors, end here #
        ##############################
        if self.regressor_col is None:
            # regardless of what args are set for these, if regressor_col is None
            # these should all be empty lists
            self._regressor_sign = list()
            self._regressor_beta_prior = list()
            self._regressor_sigma_prior = list()

            return

        def _validate(regression_params, valid_length):
            for p in regression_params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

        # regressor defaults
        num_of_regressors = len(self.regressor_col)

        _validate(
            [self.regressor_sign, self.regressor_beta_prior, self.regressor_sigma_prior],
            num_of_regressors
        )

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * num_of_regressors

        if self.regressor_beta_prior is None:
            self._regressor_beta_prior = [DEFAULT_REGRESSOR_BETA] * num_of_regressors

        if self.regressor_sigma_prior is None:
            self._regressor_sigma_prior = [DEFAULT_REGRESSOR_SIGMA] * num_of_regressors

    def _set_regression_penalty(self):
        regression_penalty = self.regression_penalty
        self._regression_penalty = getattr(constants.RegressionPenalty, regression_penalty).value

    def _set_static_regression_attributes(self):
        # if no regressors, end here
        if self.regressor_col is None:
            return

        # inside *.stan files, we need to distinguish regular regressors from positive regressors
        for index, reg_sign in enumerate(self._regressor_sign):
            if reg_sign == '+':
                self._num_of_positive_regressors += 1
                self._positive_regressor_col.append(self.regressor_col[index])
                self._positive_regressor_beta_prior.append(self._regressor_beta_prior[index])
                self._positive_regressor_sigma_prior.append(self._regressor_sigma_prior[index])
            else:
                self._num_of_regular_regressors += 1
                self._regular_regressor_col.append(self.regressor_col[index])
                self._regular_regressor_beta_prior.append(self._regressor_beta_prior[index])
                self._regular_regressor_sigma_prior.append(self._regressor_sigma_prior[index])

    def _set_with_mcmc(self):
        estimator_type = self.estimator_type
        # set `_with_mcmc` attribute based on estimator type
        # if no attribute for _is_mcmc_estimator, default to False
        if getattr(estimator_type, '_is_mcmc_estimator', False):
            self._with_mcmc = 1

    def _set_static_data_attributes(self):
        """model data input based on args at instatiation or computed from args at instantiation"""
        self._set_default_base_args()
        self._set_regression_penalty()
        self._set_static_regression_attributes()
        self._set_with_mcmc()
        self._set_init_values()

    def _validate_supported_estimator_type(self):
        if self.estimator_type not in self._supported_estimator_types:
            msg_template = "Model class: {} is incompatible with Estimator: {}"
            model_class = type(self)
            estimator_type = self.estimator_type
            raise IllegalArgument(msg_template.format(model_class, estimator_type))

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

        # validate regression columns
        if self.regressor_col is not None and \
                not set(self.regressor_col).issubset(df_columns):
            raise ModelException(
                "DataFrame does not contain specified regressor colummn(s)."
            )

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ModelException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def _set_regressor_matrix(self, df):
        # init of regression matrix depends on length of response vector
        self._positive_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)
        self._regular_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self._num_of_positive_regressors > 0:
            self._positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col,).values

        if self._num_of_regular_regressors > 0:
            self._regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col,).values

    def _set_init_values(self):
        """Set Stan init as a callable

        See: https://pystan.readthedocs.io/en/latest/api.htm
        """
        def init_values_function(seasonality):
            init_values = dict()
            if seasonality > 1:
                seas_init = np.random.normal(loc=0, scale=0.05, size=seasonality - 1)
                # catch cases with extreme values
                seas_init[seas_init > 1.0] = 1.0
                seas_init[seas_init < -1.0] = -1.0
                init_values['init_sea'] = seas_init

            return init_values

        seasonality = self._seasonality

        # init_values_partial = partial(init_values_callable, seasonality=seasonality)
        # partialfunc does not work when passed to PyStan because PyStan uses
        # inspect.getargspec(func) which seems to raise an exception with keyword-only args
        # caused by using partialfunc
        # lambda as an alternative workaround
        if seasonality > 1:
            init_values_callable = lambda: init_values_function(seasonality)  # noqa
            self._init_values = init_values_callable

    def _set_dynamic_data_attributes(self, df):
        """Stan data input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()

        self._validate_training_df(df)
        self._set_training_df_meta(df)

        # a few of the following are related with training data.
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)

        self._cauchy_sd = max(self._response) / 30.0

        self._set_regressor_matrix(df)  # depends on _num_of_observations
        self._set_init_values()
        # TODO: maybe useful to calculate vanlia (adjusted) R-Squared
        # self._adjust_smoothing_with_regression()

    def _set_model_param_names(self):
        """Model parameters to extract from Stan"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]

        # append seasonality param names
        if self._seasonality > 1:
            self._model_param_names += [param.value for param in constants.SeasonalitySamplingParameters]

        # append positive regressors if any
        if self._num_of_positive_regressors > 0:
            self._model_param_names += [
                constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value]

        # append regular regressors if any
        if self._num_of_regular_regressors > 0:
            self._model_param_names += [
                constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value]

    def _get_model_param_names(self):
        return self._model_param_names

    def _set_model_data_input(self):
        """Collects data attributes into a dict for `StanModel.sampling`"""
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

    def is_fitted(self):
        # if empty dict false, else true
        return bool(self._posterior_samples)

    @staticmethod
    def _concat_regression_coefs(pr_beta=None, rr_beta=None):
        """Concatenates regression posterior matrix

        In the case that `pr_beta` or `rr_beta` is a 1d tensor, transform to 2d tensor and
        concatenate.

        Args
        ----
        pr_beta : torch.tensor
            postive-value constrainted regression betas
        rr_beta : torch.tensor
            regular regression betas

        Returns
        -------
        torch.tensor
            concatenated 2d tensor of shape (1, len(rr_beta) + len(pr_beta))

        """
        regressor_beta = None
        if pr_beta is not None and rr_beta is not None:
            pr_beta = pr_beta if len(pr_beta.shape) == 2 else pr_beta.reshape(1, -1)
            rr_beta = rr_beta if len(rr_beta.shape) == 2 else rr_beta.reshape(1, -1)
            regressor_beta = torch.cat((pr_beta, rr_beta), dim=1)
        elif pr_beta is not None:
            regressor_beta = pr_beta
        elif rr_beta is not None:
            regressor_beta = rr_beta

        return regressor_beta

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

        # trend components
        slope_smoothing_factor = model.get(
            constants.BaseSamplingParameters.SLOPE_SMOOTHING_FACTOR.value)
        level_smoothing_factor = model.get(
            constants.BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        local_trend_levels = model.get(constants.BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        local_trend_slopes = model.get(constants.BaseSamplingParameters.LOCAL_TREND_SLOPES.value)
        residual_degree_of_freedom = model.get(
            constants.BaseSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM.value)
        residual_sigma = model.get(constants.BaseSamplingParameters.RESIDUAL_SIGMA.value)

        local_trend_coef = model.get(constants.BaseSamplingParameters.LOCAL_TREND_COEF.value)
        global_trend_power = model.get(constants.BaseSamplingParameters.GLOBAL_TREND_POWER.value)
        global_trend_coef = model.get(constants.BaseSamplingParameters.GLOBAL_TREND_COEF.value)
        local_global_trend_sums = model.get(
            constants.BaseSamplingParameters.LOCAL_GLOBAL_TREND_SUMS.value)

        # regression components
        pr_beta = model.get(constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value)
        rr_beta = model.get(constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value)
        regressor_beta = self._concat_regression_coefs(pr_beta, rr_beta)

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
        output_len = prediction_df_meta['df_length']

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
        # Regression Component
        ################################################################

        # calculate regression component
        if self.regressor_col is not None and len(self._regular_regressor_col) > 0:
            regressor_beta = regressor_beta.t()
            regressor_matrix = df[self.regressor_col].values
            regressor_torch = torch.from_numpy(regressor_matrix).double()
            regressor_component = torch.matmul(regressor_torch, regressor_beta)
            regressor_component = regressor_component.t()
        else:
            # regressor is always dependent with df. hence, no need to make full size
            regressor_component = torch.zeros((num_sample, output_len), dtype=torch.double)

        ################################################################
        # Seasonality Component
        ################################################################

        # calculate seasonality component
        if self._seasonality > 1:
            if full_len <= seasonality_levels.shape[1]:
                seasonality_component = seasonality_levels[:, :full_len]
            else:
                seasonality_forecast_length = full_len - seasonality_levels.shape[1]
                seasonality_forecast_matrix \
                    = torch.zeros((num_sample, seasonality_forecast_length), dtype=torch.double)
                seasonality_component = torch.cat(
                    (seasonality_levels, seasonality_forecast_matrix), dim=1)
        else:
            seasonality_component = torch.zeros((num_sample, full_len), dtype=torch.double)

        ################################################################
        # Trend Component
        ################################################################

        # calculate level component.
        # However, if predicted end of period > training period, update with out-of-samples forecast
        if full_len <= trained_len:
            trend_component = local_global_trend_sums[:, :full_len]

            # in-sample error are iids
            if include_error:
                error_value = nct.rvs(
                    df=residual_degree_of_freedom.unsqueeze(-1),
                    nc=0,
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=(num_sample, full_len)
                )

                error_value = torch.from_numpy(error_value.reshape(num_sample, full_len)).double()
                trend_component += error_value
        else:
            trend_component = local_global_trend_sums
            # in-sample error are iids
            if include_error:
                error_value = nct.rvs(
                    df=residual_degree_of_freedom.unsqueeze(-1),
                    nc=0,
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=(num_sample, local_global_trend_sums.shape[1])
                )

                error_value = torch.from_numpy(
                    error_value.reshape(num_sample, local_global_trend_sums.shape[1])).double()
                trend_component += error_value

            trend_forecast_matrix = torch.zeros((num_sample, n_forecast_steps), dtype=torch.double)
            trend_component = torch.cat((trend_component, trend_forecast_matrix), dim=1)

            last_local_trend_level = local_trend_levels[:, -1]
            last_local_trend_slope = local_trend_slopes[:, -1]

            trend_component_zeros = torch.zeros_like(trend_component[:, 0])

            for idx in range(trained_len, full_len):
                current_local_trend = local_trend_coef.flatten() * last_local_trend_slope
                global_trend_power_term = torch.pow(
                    torch.abs(last_local_trend_level),
                    global_trend_power.flatten()
                )
                current_global_trend = global_trend_coef.flatten() * global_trend_power_term
                trend_component[:, idx] \
                    = last_local_trend_level + current_local_trend + current_global_trend

                if include_error:
                    error_value = nct.rvs(
                        df=residual_degree_of_freedom,
                        nc=0,
                        loc=0,
                        scale=residual_sigma,
                        size=num_sample
                    )
                    error_value = torch.from_numpy(error_value).double()
                    trend_component[:, idx] += error_value

                # a 2d tensor of size (num_sample, 2) in which one of the elements
                # is always zero. We can use this to torch.max() across the sample dimensions
                trend_component_augmented = torch.cat(
                    (trend_component[:, idx][:, None], trend_component_zeros[:, None]), dim=1)

                max_value, _ = torch.max(trend_component_augmented, dim=1)

                trend_component[:, idx] = max_value

                new_local_trend_level = \
                    level_smoothing_factor * trend_component[:, idx] \
                    + (1 - level_smoothing_factor) * last_local_trend_level

                last_local_trend_slope = \
                    slope_smoothing_factor * (new_local_trend_level - last_local_trend_level) \
                    + (1 - slope_smoothing_factor) * last_local_trend_slope

                if self._seasonality > 1 and idx + self._seasonality < full_len:
                    seasonality_component[:, idx + self._seasonality] = \
                        seasonality_smoothing_factor.flatten() \
                        * (trend_component[:, idx] + seasonality_component[:, idx] -
                           new_local_trend_level) \
                        + (1 - seasonality_smoothing_factor.flatten()) * seasonality_component[:, idx]

                last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = trend_component[:, start:]
        seasonality_component = seasonality_component[:, start:]

        # sum components
        pred_array = trend_component + seasonality_component + regressor_component


        pred_array = pred_array.numpy()
        trend_component = trend_component.numpy()
        seasonality_component = seasonality_component.numpy()
        regressor_component = regressor_component.numpy()

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend_component,
                'seasonality': seasonality_component,
                'regression': regressor_component
            }

            return decomp_dict

        return {'prediction': pred_array}

    def _prepend_date_column(self, predicted_df, input_df):
        """Prepends date column from `input_df` to `predicted_df`"""

        other_cols = list(predicted_df.columns)

        # add date column
        predicted_df[self.date_col] = input_df[self.date_col].reset_index(drop=True)

        # re-order columns so date is first
        col_order = [self.date_col] + other_cols
        predicted_df = predicted_df[col_order]

        return predicted_df

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

    def get_regression_coefs(self, aggregate_method):
        """Return DataFrame regression coefficients

        If PredictMethod is `full` return `mean` of coefficients instead
        """
        # init dataframe
        reg_df = pd.DataFrame()

        # end if no regressors
        if self._num_of_regular_regressors + self._num_of_positive_regressors == 0:
            return reg_df

        pr_beta = self._aggregate_posteriors\
            .get(aggregate_method)\
            .get(constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value)

        rr_beta = self._aggregate_posteriors\
            .get(aggregate_method)\
            .get(constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value)

        # because `_conccat_regression_coefs` operates on torch tensors
        pr_beta = torch.from_numpy(pr_beta) if pr_beta is not None else pr_beta
        rr_beta = torch.from_numpy(rr_beta) if rr_beta is not None else rr_beta

        regressor_betas = self._concat_regression_coefs(pr_beta, rr_beta)

        # get column names
        pr_cols = self._positive_regressor_col
        rr_cols = self._regular_regressor_col

        # note ordering here is not the same as `self.regressor_cols` because positive
        # and negative do not have to be grouped on input
        regressor_cols = pr_cols + rr_cols

        # same note
        regressor_signs \
            = ["Positive"] * self._num_of_positive_regressors \
            + ["Regular"] * self._num_of_regular_regressors

        reg_df[COEFFICIENT_DF_COLS.REGRESSOR] = regressor_cols
        reg_df[COEFFICIENT_DF_COLS.REGRESSOR_SIGN] = regressor_signs
        reg_df[COEFFICIENT_DF_COLS.COEFFICIENT] = regressor_betas.flatten()

        return reg_df


class LGTFull(BaseLGT):
    """Concrete LGT model for full prediction

    In full prediction, the prediction occurs as a function of each parameter posterior sample,
    and the prediction results are aggregated after prediction. Prediction will
    always return the median (aka 50th percentile) along with any additional percentiles that
    are specified.

    Parameters
    ----------
    n_bootstrap_draws : int
        Number of bootstrap samples to draw from the initial MCMC or VI posterior samples.
        If None, use the original posterior draws.
    prediction_percentiles : list
        List of integers of prediction percentiles that should be returned on prediction.
    kwargs
        Additional args to pass to parent classes.

    """
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI]

    def __init__(self, n_bootstrap_draws=None, prediction_percentiles=None, **kwargs):
        # todo: assert compatible estimator
        super().__init__(**kwargs)
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles

        # set default args
        self._prediction_percentiles = None
        self._n_bootstrap_draws = self.n_bootstrap_draws
        self._set_default_args()

        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _set_default_args(self):
        if not self.prediction_percentiles:
            self._prediction_percentiles = list()
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        if not self.n_bootstrap_draws:
            self._n_bootstrap_draws = -1

    def _bootstrap(self, n):
        """Draw `n` number of bootstrap samples from the posterior_samples.

        Args
        ----
        n : int
            The number of bootstrap samples to draw

        """
        num_samples = self.estimator.num_sample
        posterior_samples = self._posterior_samples

        if n < 2:
            raise IllegalArgument("Error: The number of bootstrap draws must be at least 2")

        sample_idx = np.random.choice(
            range(num_samples),
            size=n,
            replace=True
        )

        bootstrap_samples_dict = {}
        for k, v in posterior_samples.items():
            bootstrap_samples_dict[k] = v[sample_idx]

        return bootstrap_samples_dict

    def _aggregate_full_predictions(self, array, label, percentiles):
        """Aggregates the mcmc prediction to a point estimate

        Args
        ----
        array: np.ndarray
            A 2d numpy array of shape (`num_samples`, prediction df length)
        percentiles: list
            A sorted list of one or three percentile(s) which will be used to aggregate lower, mid and upper values
        label: str
            A string used for labeling output dataframe columns
        Returns
        -------
        pd.DataFrame
            The aggregated across mcmc samples with columns for `50` aka median
            and all other percentiles specified in `percentiles`.
        """

        aggregated_array = np.percentile(array, percentiles, axis=0)
        if len(percentiles) == 1:
            aggregate_df = pd.DataFrame(aggregated_array.T, columns=[label])
        elif len(percentiles) == 3:
            aggregate_df = pd.DataFrame(aggregated_array.T, columns=[label + "_lower", label, label + "_upper"])
        else:
            raise PredictionException("Invalid input percentiles.")

        return aggregate_df

    def predict(self, df, decompose=False):
        """Return model predictions as a function of fitted model and df"""
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(self._n_bootstrap_draws) \
            if self._n_bootstrap_draws > 1 \
            else self._posterior_samples

        predicted_dict = self._predict(
            posterior_estimates=posterior_samples,
            df=df,
            include_error=True,
            decompose=decompose,
        )

        # MUST copy, or else instance var persists in memory
        percentiles = copy(self._prediction_percentiles)
        percentiles += [50]  # always find median
        percentiles = list(set(percentiles))  # unique set
        percentiles.sort()

        for k, v in predicted_dict.items():
            predicted_dict[k] = self._aggregate_full_predictions(
                array=v,
                label=k,
                percentiles=percentiles,
            )

        aggregated_df = pd.concat(predicted_dict, axis=1)
        aggregated_df.columns = aggregated_df.columns.droplevel()
        aggregated_df = self._prepend_date_column(aggregated_df, df)
        return aggregated_df

    def get_regression_coefs(self, aggregate_method='mean'):
        self._set_aggregate_posteriors()
        return super().get_regression_coefs(aggregate_method=aggregate_method)


class LGTAggregated(BaseLGT):
    """Concrete LGT model for aggregated posterior prediction

    In aggregated prediction, the parameter posterior samples are reduced using `aggregate_method`
    before performing a single prediction.

    Parameters
    ----------
    aggregate_method : { 'mean', 'median' }
        Method used to reduce parameter posterior samples

    """
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI]

    def __init__(self, aggregate_method='mean', **kwargs):
        super().__init__(**kwargs)
        self.aggregate_method = aggregate_method

        self._validate_aggregate_method()

        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _validate_aggregate_method(self):
        if self.aggregate_method not in list(self._aggregate_posteriors.keys()):
            raise PredictionException("No aggregate method defined for: `{}`".format(self.aggregate_method))

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        super().fit(df)
        self._set_aggregate_posteriors()

    def predict(self, df, decompose=False):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(self.aggregate_method)

        predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors,
            df=df,
            include_error=False,
            decompose=decompose
        )

        # must flatten to convert to DataFrame
        for k, v in predicted_dict.items():
            predicted_dict[k] = v.flatten()

        predicted_df = pd.DataFrame(predicted_dict)
        predicted_df = self._prepend_date_column(predicted_df, df)

        return predicted_df

    def get_regression_coefs(self):
        return super().get_regression_coefs(aggregate_method=self.aggregate_method)


class LGTMAP(BaseLGT):
    """Concrete LGT model for MAP (Maximum a Posteriori) prediction

    Similar to `LGTAggregated` but predition is based on Maximum a Posteriori (aka Mode)
    of the posterior.

    This model only supports MAP estimating `estimator_type`s

    """
    _supported_estimator_types = [StanEstimatorMAP, PyroEstimatorMAP]

    def __init__(self, estimator_type=StanEstimatorMAP, **kwargs):
        super().__init__(estimator_type=estimator_type, **kwargs)

        # override init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MAP.value: dict(),
        }

        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _set_map_posterior(self):
        posterior_samples = self._posterior_samples

        map_posterior = {}
        for param_name in self._model_param_names:
            param_array = posterior_samples[param_name]
            # add dimension so it works with vector math in `_predict`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update(
                {param_name: param_array}
            )

        self._aggregate_posteriors[PredictMethod.MAP.value] = map_posterior

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        super().fit(df)
        self._set_map_posterior()

    def predict(self, df, decompose=False):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(PredictMethod.MAP.value)

        predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors,
            df=df,
            include_error=False,
            decompose=decompose
        )

        # must flatten to convert to DataFrame
        for k, v in predicted_dict.items():
            predicted_dict[k] = v.flatten()

        predicted_df = pd.DataFrame(predicted_dict)
        predicted_df = self._prepend_date_column(predicted_df, df)

        return predicted_df

    def get_regression_coefs(self):
        return super().get_regression_coefs(aggregate_method=PredictMethod.MAP.value)
