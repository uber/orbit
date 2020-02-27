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
    # ---------- LGT Global Trend ---------- #
    GLOBAL_TREND_COEF_MIN = 'GT_COEF_MIN'
    GLOBAL_TREND_COEF_MAX = 'GT_COEF_MAX'
    GLOBAL_TREND_POW_MIN = 'GT_POW_MIN'
    GLOBAL_TREND_POW_MAX = 'GT_POW_MAX'
    LOCAL_TREND_COEF_MIN = 'LT_COEF_MIN'
    LOCAL_TREND_COEF_MAX = 'LT_COEF_MAX'
    # ---------- Common Local Trend ---------- #
    LEVEL_SMOOTHING_MIN = 'LEV_SM_MIN'
    LEVEL_SMOOTHING_MAX = 'LEV_SM_MAX'
    SLOPE_SMOOTHING_MIN = 'SLP_SM_MIN'
    SLOPE_SMOOTHING_MAX = 'SLP_SM_MAX'
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
