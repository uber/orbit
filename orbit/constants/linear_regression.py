from enum import Enum


class DataInputMapper(Enum):
    """Stan data inputs"""
    # ----------  Data Input ---------- #
    # observation related
    _NUM_OF_OBSERVATIONS = 'NUM_OF_OBS'
    _RESPONSE = 'RESPONSE'
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
    OBS_SIGMA = 'obs_sigma'


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
