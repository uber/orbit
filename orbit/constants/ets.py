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
    _RESPONSE_SD = 'RESPONSE_SD'
    # ---------- Seasonality ---------- #
    _SEASONALITY = 'SEASONALITY'
    _SEASONALITY_SM_INPUT = 'SEA_SM_INPUT'
    # ---------- Common Local Trend ---------- #
    _LEVEL_SM_INPUT = 'LEV_SM_INPUT'
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
    LEVEL_SMOOTHING_FACTOR = 'lev_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'


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
    # POSITIVE_REGRESSOR_BETA = 'pr_beta'
    # REGULAR_REGRESSOR_BETA = 'rr_beta'
    REGRESSION_COEFFICIENTS = 'beta'


class RegressionPenalty(Enum):
    fixed_ridge = 0
    lasso = 1
    auto_ridge = 2
