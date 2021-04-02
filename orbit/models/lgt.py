import pandas as pd
import numpy as np
from scipy.stats import nct
import torch
from copy import deepcopy
import warnings

from ..constants import lgt as constants
from ..constants.constants import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA,
    COEFFICIENT_DF_COLS,
    PredictMethod
)

from ..models.ets import BaseETS
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from ..estimators.pyro_estimator import PyroEstimatorVI, PyroEstimatorMAP
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..initializer.lgt import LGTInitializer
from ..utils.general import is_ordered_datetime
from ..models.ets import ETSMAP, ETSFull, ETSAggregated

warnings.simplefilter('always', PendingDeprecationWarning)


class BaseLGT(BaseETS):
    """Base LGT model object with shared functionality for Full, Aggregated, and MAP methods

    Parameters
    ----------
    regressor_col : list
        Names of regressor columns, if any
    regressor_sign :  list
        list with values { '+', '-', '=' } such that
        '+' indicates regressor coefficient estimates are constrained to [0, inf).
        '-' indicates regressor coefficient estimates are constrained to (-inf, 0].
        '=' indicates regressor coefficient estimates can be any value between (-inf, inf).
        The length of `regressor_sign` must be the same length as `regressor_col`. If None,
        all elements of list will be set to '='.
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
    slope_sm_input : float
        float value between [0, 1]. A larger value puts more weight on the current slope.
        If None, the model will estimate this value.
    See Also
    --------
    :class: `~orbit.models.ets.BaseETS`
    """
    _data_input_mapper = constants.DataInputMapper
    _model_name = 'lgt'

    def __init__(self, regressor_col=None, regressor_sign=None,
                 regressor_beta_prior=None, regressor_sigma_prior=None,
                 regression_penalty='fixed_ridge', lasso_scale=0.5, auto_ridge_scale=0.5,
                 slope_sm_input=None,
                 **kwargs):
        # introduce extra parameters
        self._min_nu = 5.
        self._max_nu = 40.

        self.slope_sm_input = slope_sm_input
        if regressor_col:
            warnings.warn("Regression for LGT model will be deprecated in next version, please use DLT instead",
                          PendingDeprecationWarning
                          )
        self.regressor_col = regressor_col
        self.regressor_sign = regressor_sign
        self.regressor_beta_prior = regressor_beta_prior
        self.regressor_sigma_prior = regressor_sigma_prior
        self.regression_penalty = regression_penalty
        self.lasso_scale = lasso_scale
        self.auto_ridge_scale = auto_ridge_scale

        # set private var to arg value
        # if None set default in _set_default_base_args()
        self._slope_sm_input = self.slope_sm_input

        self._regressor_sign = self.regressor_sign
        self._regressor_beta_prior = self.regressor_beta_prior
        self._regressor_sigma_prior = self.regressor_sigma_prior

        self._regression_penalty = None
        self._num_of_regressors = 0
        self._regressor_col = list()

        # positive regressors
        self._num_of_positive_regressors = 0
        self._positive_regressor_col = list()
        self._positive_regressor_beta_prior = list()
        self._positive_regressor_sigma_prior = list()
        # negative regressors
        self._num_of_negative_regressors = 0
        self._negative_regressor_col = list()
        self._negative_regressor_beta_prior = list()
        self._negative_regressor_sigma_prior = list()
        # regular regressors
        self._num_of_regular_regressors = 0
        self._regular_regressor_col = list()
        self._regular_regressor_beta_prior = list()
        self._regular_regressor_sigma_prior = list()

        # init dynamic data attributes
        # the following are set by `_set_dynamic_data_attributes()` and generally set during fit()
        # from input df
        # response data
        self._response = None
        self._num_of_observations = None
        self._cauchy_sd = None

        # regression data
        self._regular_regressor_matrix = None
        self._positive_regressor_matrix = None
        self._negative_regressor_matrix = None

        # order matters and super constructor called after attributes are set
        # since we override _set_static_data_attributes()
        super().__init__(**kwargs)

    def _set_init_values(self):
        """Set init as a callable (for Stan ONLY)
        See: https://pystan.readthedocs.io/en/latest/api.htm
        Overriding :func: `~orbit.models.BaseETS._set_init_values`
        """
        # init_values_partial = partial(init_values_callable, seasonality=seasonality)
        # partialfunc does not work when passed to PyStan because PyStan uses
        # inspect.getargspec(func) which seems to raise an exception with keyword-only args
        # caused by using partialfunc
        # lambda as an alternative workaround
        if self._seasonality > 1 or self._num_of_regressors > 0:
            init_values_callable = LGTInitializer(
                self._seasonality, self._num_of_positive_regressors, self._num_of_negative_regressors,
                self._num_of_regular_regressors
            )
            self._init_values = init_values_callable

    def _set_additional_trend_attributes(self):
        """Set additional trend attributes
        """
        if self.slope_sm_input is None:
            self._slope_sm_input = -1

    def _set_regression_default_attributes(self):
        """set and validate regression related default attributes.
        """
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
        self._num_of_regressors = len(self.regressor_col)

        _validate(
            [self.regressor_sign, self.regressor_beta_prior, self.regressor_sigma_prior],
            self._num_of_regressors
        )

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * self._num_of_regressors

        if self.regressor_beta_prior is None:
            self._regressor_beta_prior = [DEFAULT_REGRESSOR_BETA] * self._num_of_regressors

        if self.regressor_sigma_prior is None:
            self._regressor_sigma_prior = [DEFAULT_REGRESSOR_SIGMA] * self._num_of_regressors

    def _set_regression_penalty(self):
        """set and validate regression penalty related attributes.
        """
        regression_penalty = self.regression_penalty
        self._regression_penalty = getattr(constants.RegressionPenalty, regression_penalty).value

    def _set_static_regression_attributes(self):
        """set and validate regression related attributes.
        """
        # if no regressors, end here
        if self.regressor_col is None:
            return

        # inside *.stan files, we need to distinguish regular, positive and negative regressors
        for index, reg_sign in enumerate(self._regressor_sign):
            if reg_sign == '+':
                self._num_of_positive_regressors += 1
                self._positive_regressor_col.append(self.regressor_col[index])
                self._positive_regressor_beta_prior.append(self._regressor_beta_prior[index])
                self._positive_regressor_sigma_prior.append(self._regressor_sigma_prior[index])
            elif reg_sign == '-':
                self._num_of_negative_regressors += 1
                self._negative_regressor_col.append(self.regressor_col[index])
                self._negative_regressor_beta_prior.append(self._regressor_beta_prior[index])
                self._negative_regressor_sigma_prior.append(self._regressor_sigma_prior[index])
            else:
                self._num_of_regular_regressors += 1
                self._regular_regressor_col.append(self.regressor_col[index])
                self._regular_regressor_beta_prior.append(self._regressor_beta_prior[index])
                self._regular_regressor_sigma_prior.append(self._regressor_sigma_prior[index])

        self._regressor_col = self._positive_regressor_col + self._negative_regressor_col + \
                              self._regular_regressor_col

    def _set_static_data_attributes(self):
        """Cast data to the proper type mostly to match Stan required static data types
        Notes
        -----
        Overriding :func: `~orbit.models.BaseETS._set_static_data_attributes`
        It sets additional required attributes related to trend and regression
        """
        super()._set_static_data_attributes()
        self._set_additional_trend_attributes()
        self._set_regression_default_attributes()
        self._set_regression_penalty()
        self._set_static_regression_attributes()

    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api
        Notes
        -----
        Overriding :func: `~orbit.models.BaseETS._set_model_param_names`
        It sets additional required attributes related to trend and regression
        """
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]

        # append seasonality param names
        if self._seasonality > 1:
            self._model_param_names += [param.value for param in constants.SeasonalitySamplingParameters]

        # append positive regressors if any
        if self._num_of_regressors > 0:
            self._model_param_names += [
                constants.RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value]

    def _validate_training_df_with_regression(self, df):
        df_columns = df.columns
        # validate regression columns
        if self.regressor_col is not None and \
                not set(self.regressor_col).issubset(df_columns):
            raise ModelException(
                "DataFrame does not contain specified regressor column(s)."
            )

    def _set_regressor_matrix(self, df):
        """Set regressor matrix based on the input data-frame.
        Notes
        -----
        In case of absence of regression, they will be set to np.array with dim (num_of_obs, 0) to fit Stan requirement
        """
        # init of regression matrix depends on length of response vector
        self._positive_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)
        self._negative_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)
        self._regular_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self._num_of_positive_regressors > 0:
            self._positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col, ).values

        if self._num_of_negative_regressors > 0:
            self._negative_regressor_matrix = df.filter(
                items=self._negative_regressor_col, ).values

        if self._num_of_regular_regressors > 0:
            self._regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col, ).values

    def _set_dynamic_data_attributes(self, df):
        """Set required input based on input DataFrame, rather than at object instantiation.  It also set
        additional required attributes for LGT"""
        super()._validate_training_df(df)
        super()._set_training_df_meta(df)

        # set the rest of attributes related to training df
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)
        # scalar value is suggested by the author of Rlgt
        self._cauchy_sd = max(self._response) / 30.0

        # extra settings for regression
        self._validate_training_df_with_regression(df)
        self._set_regressor_matrix(df)  # depends on _num_of_observations

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
        regressor_beta = model.get(constants.RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value)

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
        if self.regressor_col is not None and len(self.regressor_col) > 0:
            regressor_beta = regressor_beta.t()
            regressor_matrix = df[self._regressor_col].values
            regressor_torch = torch.from_numpy(regressor_matrix).double()
            regression = torch.matmul(regressor_torch, regressor_beta)
            regression = regression.t()
        else:
            # regressor is always dependent with df. hence, no need to make full size
            regression = torch.zeros((num_sample, output_len), dtype=torch.double)

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
        pred_array = trend_component + seasonal_component + regression

        pred_array = pred_array.numpy()
        trend_component = trend_component.numpy()
        seasonal_component = seasonal_component.numpy()
        regression = regression.numpy()

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend_component,
                'seasonality': seasonal_component,
                'regression': regression
            }

            return decomp_dict

        return {'prediction': pred_array}

    def _get_regression_coefs(self, aggregate_method):
        """Return DataFrame regression coefficients

        If PredictMethod is `full` return `mean` of coefficients instead
        """
        # init dataframe
        coef_df = pd.DataFrame()

        # end if no regressors
        if self._num_of_regressors == 0:
            return coef_df

        coef = self._aggregate_posteriors \
            .get(aggregate_method) \
            .get(constants.RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value)

        # get column names
        pr_cols = self._positive_regressor_col
        nr_cols = self._negative_regressor_col
        rr_cols = self._regular_regressor_col

        # note ordering here is not the same as `self.regressor_cols` because positive
        # and negative do not have to be grouped on input
        regressor_cols = pr_cols + nr_cols + rr_cols

        # same note
        regressor_signs = ["Positive"] * self._num_of_positive_regressors + \
                          ["Negative"] * self._num_of_negative_regressors + \
                          ["Regular"] * self._num_of_regular_regressors

        coef_df[COEFFICIENT_DF_COLS.REGRESSOR] = regressor_cols
        coef_df[COEFFICIENT_DF_COLS.REGRESSOR_SIGN] = regressor_signs
        coef_df[COEFFICIENT_DF_COLS.COEFFICIENT] = coef.flatten()

        return coef_df


class LGTMAP(BaseLGT, ETSMAP):
    """Concrete LGT model for MAP (Maximum a Posteriori) prediction

    Similar to :class: `~orbit.models.LGTAggregated` but prediction is based on Maximum a Posteriori (aka Mode)
    of the posterior.

    This model only supports MAP estimating `estimator_type`s

    """
    _supported_estimator_types = [StanEstimatorMAP, PyroEstimatorMAP]

    def get_regression_coefs(self):
        return super()._get_regression_coefs(aggregate_method=PredictMethod.MAP.value)


class LGTFull(BaseLGT, ETSFull):
    """Concrete LGT model for full Bayesian prediction"""
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_regression_coefs(self, aggregate_method='mean'):
        self._set_aggregate_posteriors()
        return super()._get_regression_coefs(aggregate_method=aggregate_method)


class LGTAggregated(BaseLGT, ETSAggregated):
    """Concrete LGT model for aggregated posterior prediction"""
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_regression_coefs(self):
        self._set_aggregate_posteriors()
        return super()._get_regression_coefs(aggregate_method=self.aggregate_method)


