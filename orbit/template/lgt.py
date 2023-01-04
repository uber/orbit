import numpy as np
import pandas as pd
from scipy.stats import nct
from copy import deepcopy
import torch
from enum import Enum
import warnings

from ..constants.constants import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA,
    COEFFICIENT_DF_COLS,
    PredictMethod,
    PredictionKeys,
    TrainingMetaKeys,
    PredictionMetaKeys,
)
from ..exceptions import IllegalArgument, ModelException, DataInputException

# from .model_template import ModelTemplate
from .ets import ETSModel
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP
from ..estimators.pyro_estimator import PyroEstimatorSVI


class DataInputMapper(Enum):
    """
    mapping from object input to sampler
    """

    # ---------- Seasonality ---------- #
    _SEASONALITY = "SEASONALITY"
    SEASONALITY_SD = "SEASONALITY_SD"
    _SEASONALITY_SM_INPUT = "SEA_SM_INPUT"
    # ---------- Common Local Trend ---------- #
    _LEVEL_SM_INPUT = "LEV_SM_INPUT"
    _SLOPE_SM_INPUT = "SLP_SM_INPUT"
    # ----------  Noise Distribution  ---------- #
    MIN_NU = "MIN_NU"
    MAX_NU = "MAX_NU"
    CAUCHY_SD = "CAUCHY_SD"
    # ----------  Regressions ---------- #
    NUM_OF_POSITIVE_REGRESSORS = "NUM_OF_PR"
    POSITIVE_REGRESSOR_MATRIX = "PR_MAT"
    POSITIVE_REGRESSOR_BETA_PRIOR = "PR_BETA_PRIOR"
    POSITIVE_REGRESSOR_SIGMA_PRIOR = "PR_SIGMA_PRIOR"
    NUM_OF_NEGATIVE_REGRESSORS = "NUM_OF_NR"
    NEGATIVE_REGRESSOR_MATRIX = "NR_MAT"
    NEGATIVE_REGRESSOR_BETA_PRIOR = "NR_BETA_PRIOR"
    NEGATIVE_REGRESSOR_SIGMA_PRIOR = "NR_SIGMA_PRIOR"
    NUM_OF_REGULAR_REGRESSORS = "NUM_OF_RR"
    REGULAR_REGRESSOR_MATRIX = "RR_MAT"
    REGULAR_REGRESSOR_BETA_PRIOR = "RR_BETA_PRIOR"
    REGULAR_REGRESSOR_SIGMA_PRIOR = "RR_SIGMA_PRIOR"
    _REGRESSION_PENALTY = "REG_PENALTY_TYPE"
    AUTO_RIDGE_SCALE = "AUTO_RIDGE_SCALE"
    LASSO_SCALE = "LASSO_SCALE"
    # handle missing values
    IS_VALID_RESPONSE = "IS_VALID_RES"


class BaseSamplingParameters(Enum):
    """
    base parameters in posteriors sampling
    """

    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = "l"
    LOCAL_TREND_SLOPES = "b"
    LEVEL_SMOOTHING_FACTOR = "lev_sm"
    SLOPE_SMOOTHING_FACTOR = "slp_sm"
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = "obs_sigma"
    RESIDUAL_DEGREE_OF_FREEDOM = "nu"
    # ---------- LGT Model Specific ---------- #
    LOCAL_GLOBAL_TREND_SUMS = "lgt_sum"
    GLOBAL_TREND_POWER = "gt_pow"
    LOCAL_TREND_COEF = "lt_coef"
    GLOBAL_TREND_COEF = "gt_coef"


class SeasonalitySamplingParameters(Enum):
    """
    seasonality component related parameters in posteriors sampling
    """

    SEASONALITY_LEVELS = "s"
    SEASONALITY_SMOOTHING_FACTOR = "sea_sm"


class RegressionSamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """

    REGRESSION_COEFFICIENTS = "beta"


class LatentSamplingParameters(Enum):
    """
    latent variables to be sampled
    """

    REGRESSION_POSITIVE_COEFFICIENTS = "pr_beta"
    REGRESSION_NEGATIVE_COEFFICIENTS = "nr_beta"
    REGRESSION_REGULAR_COEFFICIENTS = "rr_beta"
    INITIAL_SEASONALITY = "init_sea"


class RegressionPenalty(Enum):
    fixed_ridge = 0
    lasso = 1
    auto_ridge = 2


class LGTModel(ETSModel):
    """
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
    """

    # data labels for sampler
    _data_input_mapper = DataInputMapper
    # used to match name of `*.stan` or `*.pyro` file to look for the model
    _model_name = "lgt"
    _supported_estimator_types = [StanEstimatorMAP, StanEstimatorMCMC, PyroEstimatorSVI]

    def __init__(
        self,
        regressor_col=None,
        regressor_sign=None,
        regressor_beta_prior=None,
        regressor_sigma_prior=None,
        regression_penalty="fixed_ridge",
        lasso_scale=0.5,
        auto_ridge_scale=0.5,
        slope_sm_input=None,
        **kwargs,
    ):
        # introduce extra parameters
        self.min_nu = 5.0
        self.max_nu = 40.0

        self.slope_sm_input = slope_sm_input
        if regressor_col:
            warnings.warn(
                "Regression for LGT model will be deprecated in next version, please use DLT instead",
                PendingDeprecationWarning,
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
        self._regressor_col = list()

        self.num_of_regressors = 0
        # positive regressors
        self.num_of_positive_regressors = 0
        self.positive_regressor_col = list()
        self.positive_regressor_beta_prior = list()
        self.positive_regressor_sigma_prior = list()
        # negative regressors
        self.num_of_negative_regressors = 0
        self.negative_regressor_col = list()
        self.negative_regressor_beta_prior = list()
        self.negative_regressor_sigma_prior = list()
        # regular regressors
        self.num_of_regular_regressors = 0
        self.regular_regressor_col = list()
        self.regular_regressor_beta_prior = list()
        self.regular_regressor_sigma_prior = list()

        # init dynamic data attributes
        # the following are set by `_set_dynamic_attributes()` and generally set during fit()
        # from input df
        # response data
        self.cauchy_sd = None

        # regression data
        self.positive_regressor_matrix = None
        self.negative_regressor_matrix = None
        self.regular_regressor_matrix = None

        super().__init__(**kwargs)

    def set_init_values(self):
        """Override function from base class"""
        init_values = None
        if self._seasonality > 1 or self.num_of_regressors > 0:
            init_values = dict()
            if self._seasonality > 1:
                init_sea = np.clip(
                    np.random.normal(loc=0, scale=0.05, size=self._seasonality - 1),
                    -1.0,
                    1.0,
                )
                init_values[
                    LatentSamplingParameters.INITIAL_SEASONALITY.value
                ] = init_sea
            if self.num_of_positive_regressors > 0:
                x = np.clip(
                    np.random.normal(
                        loc=0, scale=0.1, size=self.num_of_positive_regressors
                    ),
                    1e-5,
                    2.0,
                )
                init_values[
                    LatentSamplingParameters.REGRESSION_POSITIVE_COEFFICIENTS.value
                ] = x
            if self.num_of_negative_regressors > 0:
                x = np.clip(
                    np.random.normal(
                        loc=0, scale=0.1, size=self.num_of_negative_regressors
                    ),
                    -2.0,
                    -1e-5,
                )
                init_values[
                    LatentSamplingParameters.REGRESSION_NEGATIVE_COEFFICIENTS.value
                ] = x
            if self.num_of_regular_regressors > 0:
                x = np.clip(
                    np.random.normal(
                        loc=0, scale=0.1, size=self.num_of_regular_regressors
                    ),
                    -2.0,
                    2.0,
                )
                init_values[
                    LatentSamplingParameters.REGRESSION_REGULAR_COEFFICIENTS.value
                ] = x
        self._init_values = init_values

    def _set_additional_trend_attributes(self):
        """Set additional trend attributes"""
        if self.slope_sm_input is None:
            self._slope_sm_input = -1

    def _set_regression_default_attributes(self):
        """set and validate regression related default attributes."""
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
                    raise IllegalArgument(
                        "Wrong dimension length in Regression Param Input"
                    )

        # regressor defaults
        self.num_of_regressors = len(self.regressor_col)

        _validate(
            [
                self.regressor_sign,
                self.regressor_beta_prior,
                self.regressor_sigma_prior,
            ],
            self.num_of_regressors,
        )

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * self.num_of_regressors

        if self.regressor_beta_prior is None:
            self._regressor_beta_prior = [
                DEFAULT_REGRESSOR_BETA
            ] * self.num_of_regressors

        if self.regressor_sigma_prior is None:
            self._regressor_sigma_prior = [
                DEFAULT_REGRESSOR_SIGMA
            ] * self.num_of_regressors

    def _set_regression_penalty(self):
        """set and validate regression penalty related attributes."""
        regression_penalty = self.regression_penalty
        self._regression_penalty = getattr(RegressionPenalty, regression_penalty).value

    def _set_static_regression_attributes(self):
        """set and validate regression related attributes."""
        # if no regressors, end here
        if self.regressor_col is None:
            return

        # inside *.stan files, we need to distinguish regular, positive and negative regressors
        for index, reg_sign in enumerate(self._regressor_sign):
            if reg_sign == "+":
                self.num_of_positive_regressors += 1
                self.positive_regressor_col.append(self.regressor_col[index])
                self.positive_regressor_beta_prior.append(
                    self._regressor_beta_prior[index]
                )
                self.positive_regressor_sigma_prior.append(
                    self._regressor_sigma_prior[index]
                )
            elif reg_sign == "-":
                self.num_of_negative_regressors += 1
                self.negative_regressor_col.append(self.regressor_col[index])
                self.negative_regressor_beta_prior.append(
                    self._regressor_beta_prior[index]
                )
                self.negative_regressor_sigma_prior.append(
                    self._regressor_sigma_prior[index]
                )
            else:
                self.num_of_regular_regressors += 1
                self.regular_regressor_col.append(self.regressor_col[index])
                self.regular_regressor_beta_prior.append(
                    self._regressor_beta_prior[index]
                )
                self.regular_regressor_sigma_prior.append(
                    self._regressor_sigma_prior[index]
                )

        self._regressor_col = (
            self.positive_regressor_col
            + self.negative_regressor_col
            + self.regular_regressor_col
        )

    def _set_static_attributes(self):
        """Cast data to the proper type mostly to match Stan required static data types
        Notes
        -----
        Overriding :func: `~orbit.models.BaseETS._set_static_attributes`
        It sets additional required attributes related to trend and regression
        """
        super()._set_static_attributes()
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
        self._model_param_names += [param.value for param in BaseSamplingParameters]

        # append seasonality param names
        if self._seasonality > 1:
            self._model_param_names += [
                param.value for param in SeasonalitySamplingParameters
            ]

        # append positive regressors if any
        if self.num_of_regressors > 0:
            self._model_param_names += [
                RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value
            ]

    def _validate_training_df_with_regression(self, df):
        df_columns = df.columns
        # validate regression columns
        if self.regressor_col is not None and not set(self.regressor_col).issubset(
            df_columns
        ):
            raise ModelException(
                "DataFrame does not contain specified regressor column(s)."
            )

    def _set_regressor_matrix(self, df, num_of_observations):
        """Set regressor matrix based on the input data-frame.
        Notes
        -----
        In case of absence of regression, they will be set to np.array with dim (num_of_obs, 0) to fit Stan requirement
        """
        # init of regression matrix depends on length of response vector
        self.positive_regressor_matrix = np.zeros(
            (num_of_observations, 0), dtype=np.double
        )
        self.negative_regressor_matrix = np.zeros(
            (num_of_observations, 0), dtype=np.double
        )
        self.regular_regressor_matrix = np.zeros(
            (num_of_observations, 0), dtype=np.double
        )

        # update regression matrices
        if self.num_of_positive_regressors > 0:
            self.positive_regressor_matrix = df.filter(
                items=self.positive_regressor_col,
            ).values

        if self.num_of_negative_regressors > 0:
            self.negative_regressor_matrix = df.filter(
                items=self.negative_regressor_col,
            ).values

        if self.num_of_regular_regressors > 0:
            self.regular_regressor_matrix = df.filter(
                items=self.regular_regressor_col,
            ).values

    def set_dynamic_attributes(self, df, training_meta):
        """Set required input based on input DataFrame, rather than at object instantiation.  It also set
        additional required attributes for LGT"""
        super().set_dynamic_attributes(df, training_meta)
        # scalar value is suggested by the author of Rlgt
        self.cauchy_sd = max(training_meta[TrainingMetaKeys.RESPONSE.value]) / 30.0
        if any(
            training_meta[TrainingMetaKeys.RESPONSE.value][self.is_valid_response] < 0
        ):
            raise DataInputException(
                "LGT model does not allow negative response values.."
            )

        # extra validation and settings for regression
        self._validate_training_df_with_regression(df)
        # depends on num_of_observations
        self._set_regressor_matrix(df, training_meta[TrainingMetaKeys.NUM_OF_OBS.value])

    def predict(
        self,
        posterior_estimates,
        df,
        training_meta,
        prediction_meta,
        include_error=False,
        **kwargs,
    ):
        """Vectorized version of prediction math"""
        ################################################################
        # Prediction Attributes
        ################################################################
        # n_forecast_steps = prediction_meta[PredictionMetaKeys.FUTURE_STEPS.value]
        start = prediction_meta[PredictionMetaKeys.START_INDEX.value]
        trained_len = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        output_len = prediction_meta[PredictionMetaKeys.PREDICTION_DF_LEN.value]
        full_len = prediction_meta[PredictionMetaKeys.END_INDEX.value]

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
            SeasonalitySamplingParameters.SEASONALITY_LEVELS.value
        )
        seasonality_smoothing_factor = model.get(
            SeasonalitySamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend components
        slope_smoothing_factor = model.get(
            BaseSamplingParameters.SLOPE_SMOOTHING_FACTOR.value
        )
        level_smoothing_factor = model.get(
            BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value
        )
        local_trend_levels = model.get(BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        local_trend_slopes = model.get(BaseSamplingParameters.LOCAL_TREND_SLOPES.value)
        residual_degree_of_freedom = model.get(
            BaseSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM.value
        )
        residual_sigma = model.get(BaseSamplingParameters.RESIDUAL_SIGMA.value)

        local_trend_coef = model.get(BaseSamplingParameters.LOCAL_TREND_COEF.value)
        global_trend_power = model.get(BaseSamplingParameters.GLOBAL_TREND_POWER.value)
        global_trend_coef = model.get(BaseSamplingParameters.GLOBAL_TREND_COEF.value)
        local_global_trend_sums = model.get(
            BaseSamplingParameters.LOCAL_GLOBAL_TREND_SUMS.value
        )

        # regression components
        regressor_beta = model.get(
            RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value
        )

        ################################################################
        # Regression Component
        ################################################################

        # calculate regression component
        if self.regressor_col is not None and len(self.regressor_col) > 0:
            regressor_beta = regressor_beta.t()
            if len(regressor_beta.shape) == 1:
                regressor_beta = regressor_beta.unsqueeze(0)
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
                seasonality_forecast_matrix = torch.zeros(
                    (num_sample, seasonality_forecast_length), dtype=torch.double
                )
                seasonal_component = torch.cat(
                    (seasonality_levels, seasonality_forecast_matrix), dim=1
                )
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
                    size=(num_sample, full_len),
                )

                error_value = torch.from_numpy(
                    error_value.reshape(num_sample, full_len)
                ).double()
                trend_component += error_value
        else:
            trend_forecast_length = full_len - trained_len
            trend_forecast_init = torch.zeros(
                (num_sample, trend_forecast_length), dtype=torch.double
            )
            trend_component = local_global_trend_sums
            # in-sample error are iids
            if include_error:
                error_value = nct.rvs(
                    df=residual_degree_of_freedom.unsqueeze(-1),
                    nc=0,
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=(num_sample, local_global_trend_sums.shape[1]),
                )

                error_value = torch.from_numpy(
                    error_value.reshape(num_sample, local_global_trend_sums.shape[1])
                ).double()
                trend_component += error_value

            trend_component = torch.cat((trend_component, trend_forecast_init), dim=1)

            last_local_trend_level = local_trend_levels[:, -1]
            last_local_trend_slope = local_trend_slopes[:, -1]

            trend_component_zeros = torch.zeros_like(trend_component[:, 0])

            for idx in range(trained_len, full_len):
                current_local_trend = (
                    local_trend_coef.flatten() * last_local_trend_slope
                )
                global_trend_power_term = torch.pow(
                    torch.abs(last_local_trend_level), global_trend_power.flatten()
                )
                current_global_trend = (
                    global_trend_coef.flatten() * global_trend_power_term
                )
                trend_component[:, idx] = (
                    last_local_trend_level + current_local_trend + current_global_trend
                )

                if include_error:
                    error_value = nct.rvs(
                        df=residual_degree_of_freedom,
                        nc=0,
                        loc=0,
                        scale=residual_sigma,
                        size=num_sample,
                    )
                    error_value = torch.from_numpy(error_value).double()
                    trend_component[:, idx] += error_value

                # a 2d tensor of size (num_sample, 2) in which one of the elements
                # is always zero. We can use this to torch.max() across the sample dimensions
                trend_component_augmented = torch.cat(
                    (trend_component[:, idx][:, None], trend_component_zeros[:, None]),
                    dim=1,
                )

                max_value, _ = torch.max(trend_component_augmented, dim=1)

                trend_component[:, idx] = max_value

                new_local_trend_level = (
                    level_smoothing_factor * trend_component[:, idx]
                    + (1 - level_smoothing_factor) * last_local_trend_level
                )

                last_local_trend_slope = (
                    slope_smoothing_factor
                    * (new_local_trend_level - last_local_trend_level)
                    + (1 - slope_smoothing_factor) * last_local_trend_slope
                )

                if self._seasonality > 1 and idx + self._seasonality < full_len:
                    seasonal_component[:, idx + self._seasonality] = (
                        seasonality_smoothing_factor.flatten()
                        * (
                            trend_component[:, idx]
                            + seasonal_component[:, idx]
                            - new_local_trend_level
                        )
                        + (1 - seasonality_smoothing_factor.flatten())
                        * seasonal_component[:, idx]
                    )

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

        out = {
            PredictionKeys.PREDICTION.value: pred_array,
            PredictionKeys.TREND.value: trend_component,
            PredictionKeys.SEASONALITY.value: seasonal_component,
            PredictionKeys.REGRESSION.value: regression,
        }

        return out

    def get_regression_coefs(
        self,
        training_meta,
        point_method,
        point_posteriors,
        posterior_samples,
        lower=0.05,
        upper=0.95,
    ):
        """Return DataFrame regression coefficients.
          If point_method is None when fitting, return the median of coefficients.

        Parameters
        -----------
        lower : float between (0, 1). default to be 0.05
            lower bound for the CI
        upper : float between (0, 1). default to be 0.95.
            upper bound for the CI

        Returns
        -------
        pandas data frame holding the regression coefficients
        """
        # init dataframe
        coef_df = pd.DataFrame()

        # end if no regressors
        if self.num_of_regressors == 0:
            return coef_df

        _point_method = point_method
        if point_method is None:
            _point_method = PredictMethod.MEDIAN.value

        coef = point_posteriors.get(_point_method).get(
            RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value
        )

        # get column names
        pr_cols = self.positive_regressor_col
        nr_cols = self.negative_regressor_col
        rr_cols = self.regular_regressor_col

        # note ordering here is not the same as `self.regressor_cols` because regressors here are grouped by signs
        regressor_cols = pr_cols + nr_cols + rr_cols

        # same note
        regressor_signs = (
            ["Positive"] * self.num_of_positive_regressors
            + ["Negative"] * self.num_of_negative_regressors
            + ["Regular"] * self.num_of_regular_regressors
        )

        coef_df[COEFFICIENT_DF_COLS.REGRESSOR] = regressor_cols
        coef_df[COEFFICIENT_DF_COLS.REGRESSOR_SIGN] = regressor_signs
        coef_df[COEFFICIENT_DF_COLS.COEFFICIENT] = coef.flatten()

        # if we have posteriors distribution and also include ci
        if point_method in [None, PredictMethod.MEAN.value, PredictMethod.MEDIAN.value]:
            coef_samples = posterior_samples.get(
                RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value
            )
            coef_lower = np.quantile(coef_samples, lower, axis=0)
            coef_upper = np.quantile(coef_samples, upper, axis=0)
            coef_df[COEFFICIENT_DF_COLS.COEFFICIENT + "_lower"] = coef_lower
            coef_df[COEFFICIENT_DF_COLS.COEFFICIENT + "_upper"] = coef_upper
            n_pos = np.apply_along_axis(lambda x: np.sum(x >= 0), 0, coef_samples)
            n_total = coef_samples.shape[0]
            coef_df[COEFFICIENT_DF_COLS.PROB_COEF_POS] = n_pos / n_total
            coef_df[COEFFICIENT_DF_COLS.PROB_COEF_NEG] = 1 - n_pos / n_total

        return coef_df
