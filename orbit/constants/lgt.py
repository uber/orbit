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
    SEASONALITY_SMOOTHING_LOC= 'SEA_SM_LOC'
    SEASONALITY_SMOOTHING_SHAPE = 'SEA_SM_SHAPE'
    # ---------- LGT Global Trend ---------- #
    # GLOBAL_TREND_COEF_MIN = 'GT_COEF_MIN'
    # GLOBAL_TREND_COEF_MAX = 'GT_COEF_MAX'
    # GLOBAL_TREND_POW_MIN = 'GT_POW_MIN'
    # GLOBAL_TREND_POW_MAX = 'GT_POW_MAX'
    # LOCAL_TREND_COEF_MIN = 'LT_COEF_MIN'
    # LOCAL_TREND_COEF_MAX = 'LT_COEF_MAX'
    # ---------- Common Local Trend ---------- #
    LEVEL_SMOOTHING_LOC = 'LEV_SM_LOC'
    LEVEL_SMOOTHING_SHAPE = 'LEV_SM_SHAPE'
    SLOPE_SMOOTHING_LOC = 'SLP_SM_LOC'
    SLOPE_SMOOTHING_SHAPE = 'SLP_SM_SHAPE'
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
    R_SQUARED_PENALTY = 'R_SQUARED_PENALTY'


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


class RegressionPenalty(Enum):
    fixed_ridge = 0
    lasso = 1
    auto_ridge = 2
