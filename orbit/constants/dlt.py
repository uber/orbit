from enum import Enum


class DataInputMapper(Enum):
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
    SEASONALITY_SM_INPUT = 'SEA_SM_INPUT'
    # ---------- Common Local Trend ---------- #
    LEVEL_SM_INPUT = 'LEV_SM_INPUT'
    SLOPE_SM_INPUT = 'SLP_SM_INPUT'
    # ---------- Global Trend ---------- #
    _GLOBAL_TREND_OPTION = 'GLOBAL_TREND_OPTION'
    TIME_DELTA = 'TIME_DELTA'
    # ---------- Damped Trend ---------- #
    DAMPED_FACTOR_MIN = 'DAMPED_FACTOR_MIN'
    DAMPED_FACTOR_MAX = 'DAMPED_FACTOR_MAX'
    DAMPED_FACTOR_FIXED = 'DAMPED_FACTOR_FIXED'
    # ----------  Noise Distribution  ---------- #
    MIN_NU = 'MIN_NU'
    MAX_NU = 'MAX_NU'
    CAUCHY_SD = 'CAUCHY_SD'
    # ----------  Regressions ---------- #
    NUM_OF_POSITIVE_REGRESSORS = 'NUM_OF_PR'
    POSITIVE_REGRESSOR_MATRIX = 'PR_MAT'
    POSITIVE_REGRESSOR_BETA_PRIOR = 'PR_BETA_PRIOR'
    POSITIVE_REGRESSOR_SIGMA_PRIOR = 'PR_SIGMA_PRIOR'
    NUM_OF_REGULAR_REGRESSORS = 'NUM_OF_RR'
    REGULAR_REGRESSOR_MATRIX = 'RR_MAT'
    REGULAR_REGRESSOR_BETA_PRIOR = 'RR_BETA_PRIOR'
    REGULAR_REGRESSOR_SIGMA_PRIOR = 'RR_SIGMA_PRIOR'
    _REGRESSION_PENALTY = 'REG_PENALTY_TYPE'
    AUTO_RIDGE_SCALE = 'AUTO_RIDGE_SCALE'
    LASSO_SCALE = 'LASSO_SCALE'
    # Experimental; to avoid over-parameterization of latent variable vs. regression when
    # they have similar marginal impact.  In that case, penalty kick in to reward more to explain variation with
    # regression instead of latent variables.
    # R_SQUARED_PENALTY = 'R_SQUARED_PENALTY'


class GlobalTrendOption(Enum):
    linear = 0
    loglinear = 1
    logistic = 2
    flat = 3


class BaseSamplingParameters(Enum):
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


class GlobalTrendSamplingParameters(Enum):
    GLOBAL_TREND = 'gt_sum'
    GLOBAL_TREND_SLOPE = 'gb'
    GLOBAL_TREND_LEVEL = 'gl'


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


class RegressionPenalty(Enum):
    fixed_ridge = 0
    lasso = 1
    auto_ridge = 2