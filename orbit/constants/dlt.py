from enum import Enum


class StanInputMapper(Enum):
    """
    mapping from object input to stan file
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    NUM_OF_OBSERVATIONS = 'NUM_OF_OBS'
    RESPONSE = 'RESPONSE'
    # ---------- Seasonality ---------- #
    SEASONALITY = 'SEASONALITY'
    SEASONALITY_MIN = 'SEA_MIN'
    SEASONALITY_MAX = 'SEA_MAX'
    SEASONALITY_SMOOTHING_MIN = 'SEA_SM_MIN'
    SEASONALITY_SMOOTHING_MAX = 'SEA_SM_MAX'
    # ---------- Common Local Trend ---------- #
    LEVEL_SMOOTHING_MIN = 'LEV_SM_MIN'
    LEVEL_SMOOTHING_MAX = 'LEV_SM_MAX'
    SLOPE_SMOOTHING_MIN = 'SLP_SM_MIN'
    SLOPE_SMOOTHING_MAX = 'SLP_SM_MAX'
    # ---------- Global Trend ---------- #
    USE_LOG_GLOBAL_TREND = 'USE_LOG_G_TREND'
    # ---------- Damped Trend ---------- #
    DAMPED_FACTOR_MIN = 'DAMPED_FACTOR_MIN'
    DAMPED_FACTOR_MAX = 'DAMPED_FACTOR_MAX'
    DAMPED_FACTOR_FIXED = 'DAMPED_FACTOR_FIXED'
    # ----------  Noise Distribution  ---------- #
    MIN_NU = 'MIN_NU'
    MAX_NU = 'MAX_NU'
    CAUCHY_SD = 'CAUCHY_SD'
    # ----------  Regressions ---------- #
    FIX_REGRESSION_COEF_SD = 'FIX_REG_COEF_SD'
    REGRESSOR_SIGMA_SD = 'REG_SIGMA_SD'
    REGRESSION_COEF_MAX = 'BETA_MAX'
    NUM_OF_POSITIVE_REGRESSORS = 'NUM_OF_PR'
    POSITIVE_REGRESSOR_MATRIX = 'PR_MAT'
    POSITIVE_REGRESSOR_BETA_PRIOR = 'PR_BETA_PRIOR'
    POSITIVE_REGRESSOR_SIGMA_PRIOR = 'PR_SIGMA_PRIOR'
    NUM_OF_REGULAR_REGRESSORS = 'NUM_OF_RR'
    REGULAR_REGRESSOR_MATRIX = 'RR_MAT'
    REGULAR_REGRESSOR_BETA_PRIOR = 'RR_BETA_PRIOR'
    REGULAR_REGRESSOR_SIGMA_PRIOR = 'RR_SIGMA_PRIOR'


class BaseStanSamplingParameters(Enum):
    """
    The stan output sampling parameters related with DLT base model.
    """
    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = 'l'
    LOCAL_TREND_SLOPES = 'b'
    LEVEL_SMOOTHING_FACTOR = 'lev_sm'
    SLOPE_SMOOTHING_FACTOR = 'slp_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'
    RESIDUAL_DEGREE_OF_FREEDOM = 'nu'
    # ---------- DLT Model Specific ---------- #
    LOCAL_TREND = 'lt_sum'
    GLOBAL_TREND = 'gt_sum'
    GLOBAL_TREND_SLOPE = 'gb'
    GLOBAL_TREND_LEVEL = 'gl'


# class LogGlobalTrendSamplingParameters(Enum):
#     GLOBAL_TREND_SHAPE = 'gs'


class DampedTrendStanSamplingParameters(Enum):
    """
    The optional stan output sampling parameters applied when damped factor optimization required.
    """
    DAMPED_FACTOR = 'damped_factor'


class SeasonalityStanSamplingParameters(Enum):
    """
    The stan output sampling parameters related with seasonality component.
    """
    SEASONALITY_LEVELS = 's'
    # we don't need initial seasonality
    # INITIAL_SEASONALITY = 'init_sea'
    SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'


class RegressionStanSamplingParameters(Enum):
    """
    The stan output sampling parameters related with regression component.
    """
    POSITIVE_REGRESSOR_BETA = 'pr_beta'
    REGULAR_REGRESSOR_BETA = 'rr_beta'
