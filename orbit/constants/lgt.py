from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to stan file
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    _NUM_OF_OBSERVATIONS = 'NUM_OF_OBS'
    _RESPONSE = 'RESPONSE'
    # ---------- Seasonality ---------- #
    _SEASONALITY = 'SEASONALITY'
    _SEASONALITY_SM_INPUT = 'SEA_SM_INPUT'
    # ---------- LGT Global Trend ---------- #
    # GLOBAL_TREND_COEF_MIN = 'GT_COEF_MIN'
    # GLOBAL_TREND_COEF_MAX = 'GT_COEF_MAX'
    # GLOBAL_TREND_POW_MIN = 'GT_POW_MIN'
    # GLOBAL_TREND_POW_MAX = 'GT_POW_MAX'
    # LOCAL_TREND_COEF_MIN = 'LT_COEF_MIN'
    # LOCAL_TREND_COEF_MAX = 'LT_COEF_MAX'
    # ---------- Common Local Trend ---------- #
    _LEVEL_SM_INPUT = 'LEV_SM_INPUT'
    _SLOPE_SM_INPUT = 'SLP_SM_INPUT'
    # ----------  Noise Distribution  ---------- #
    _MIN_NU = 'MIN_NU'
    _MAX_NU = 'MAX_NU'
    _CAUCHY_SD = 'CAUCHY_SD'
    # ----------  Regressions ---------- #
    _NUM_OF_POSITIVE_REGRESSORS = 'NUM_OF_PR'
    _POSITIVE_REGRESSOR_MATRIX = 'PR_MAT'
    _POSITIVE_REGRESSOR_BETA_PRIOR = 'PR_BETA_PRIOR'
    _POSITIVE_REGRESSOR_SIGMA_PRIOR = 'PR_SIGMA_PRIOR'
    _NUM_OF_REGULAR_REGRESSORS = 'NUM_OF_RR'
    _REGULAR_REGRESSOR_MATRIX = 'RR_MAT'
    _REGULAR_REGRESSOR_BETA_PRIOR = 'RR_BETA_PRIOR'
    _REGULAR_REGRESSOR_SIGMA_PRIOR = 'RR_SIGMA_PRIOR'
    _REGRESSION_PENALTY = 'REG_PENALTY_TYPE'
    AUTO_RIDGE_SCALE = 'AUTO_RIDGE_SCALE'
    LASSO_SCALE = 'LASSO_SCALE'
    _WITH_MCMC = 'WITH_MCMC'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with LGT base model.
    """
    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = 'l'
    LOCAL_TREND_SLOPES = 'b'
    LEVEL_SMOOTHING_FACTOR = 'lev_sm'
    SLOPE_SMOOTHING_FACTOR = 'slp_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'
    RESIDUAL_DEGREE_OF_FREEDOM = 'nu'
    # ---------- LGT Model Specific ---------- #
    LOCAL_GLOBAL_TREND_SUMS = 'lgt_sum'
    GLOBAL_TREND_POWER = 'gt_pow'
    LOCAL_TREND_COEF = 'lt_coef'
    GLOBAL_TREND_COEF = 'gt_coef'


class SeasonalitySamplingParameters(Enum):
    """
    The stan output sampling parameters related with seasonality component.
    """
    SEASONALITY_LEVELS = 's'
    SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'


class RegressionSamplingParameters(Enum):
    """
    The stan output sampling parameters related with regression component.
    """
    POSITIVE_REGRESSOR_BETA = 'pr_beta'
    REGULAR_REGRESSOR_BETA = 'rr_beta'


class RegressionPenalty(Enum):
    fixed_ridge = 0
    lasso = 1
    auto_ridge = 2
