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
    _NUM_OF_NEGATIVE_REGRESSORS = 'NUM_OF_NR'
    _NEGATIVE_REGRESSOR_MATRIX = 'NR_MAT'
    _NEGATIVE_REGRESSOR_BETA_PRIOR = 'NR_BETA_PRIOR'
    _NEGATIVE_REGRESSOR_SIGMA_PRIOR = 'NR_SIGMA_PRIOR'
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
    base parameters in posteriors sampling
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
    seasonality component related parameters in posteriors sampling
    """
    SEASONALITY_LEVELS = 's'
    SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'


class RegressionSamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """
    REGRESSION_COEFFICIENTS = 'beta'


class LatentSamplingParameters(Enum):
    """
    latent variables to be sampled
    """
    REGRESSION_POSITIVE_COEFFICIENTS = 'pr_beta'
    REGRESSION_NEGATIVE_COEFFICIENTS = 'nr_beta'
    REGRESSION_REGULAR_COEFFICIENTS = 'rr_beta'
    INITIAL_SEASONALITY = 'init_sea'


class RegressionPenalty(Enum):
    fixed_ridge = 0
    lasso = 1
    auto_ridge = 2
