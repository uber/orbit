from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to pyro input
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    _NUM_OF_OBSERVATIONS = 'T'
    _RESPONSE = 'RESPONSE'
    # ----------  Knots Related  ---------- #
    _NUM_KNOTS_LEVEL = 'N_KNOTS_LEV'
    _NUM_KNOTS_REGRESSOR = 'N_KNOTS_COEF'
    _KERNEL_LEVEL = 'K_LEV'
    _KERNEL_REGRESSOR = 'K_COEF'
    # ----------  Level  ---------- #
    _LEVEL_LATENT_SIGMA = 'LEV_LAT_SIGMA'
    # ----------  Regression  ---------- #
    _NUM_OF_REGULAR_REGRESSORS = 'NUM_RR'
    _NUM_OF_POSITIVE_REGRESSORS = 'NUM_PR'
    _REGULAR_REGRESSOR_MATRIX = 'RR'
    _POSITIVE_REGRESSOR_MATRIX = 'PR'
    _REGULAR_REGRESSOR_LATENT_LOC_PRIOR = 'RR_BETA_LAT_LOC'
    _REGULAR_REGRESSOR_LATENT_SCALE_PRIOR = 'RR_BETA_LAT_SCALE'
    _POSITIVE_REGRESSOR_LATENT_LOC_PRIOR = 'PR_BETA_LAT_LOC'
    _POSITIVE_REGRESSOR_LATENT_SCALE_PRIOR = 'PR_BETA_LAT_SCALE'
    _POSITIVE_REGRESSOR_STEP_SCALE_PRIOR = 'PR_BETA_STEP_SCALE'
    # ----------  Prior Specification  ---------- #
    _RESPONSE_SD = 'SDY'
    _NUM_INSERT_PRIOR = 'N_PRIOR'
    _INSERT_PRIOR_MEAN = 'PRIOR_MEAN'
    _INSERT_PRIOR_SD = 'PRIOR_SD'
    _INSERT_PRIOR_TP_IDX = 'PRIOR_TP_IDX'
    _INSERT_PRIOR_IDX = 'PRIOR_IDX'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with LGT base model.
    """
    LEVEL_LATENT = 'lev_lat'
    LEVEL = 'lev'
    YHAT = 'yhat'
    OBS_SIGMA = 'obs_sigma'

# class SeasonalitySamplingParameters(Enum):
#     """
#     The stan output sampling parameters related with seasonality component.
#     """
#     SEASONALITY_LEVELS = 's'
#     SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'


class RegressionSamplingParameters(Enum):
    """
    The stan output sampling parameters related with regression component.
    """

    # REGULAR_REGRESSOR_LATENT_BETA = 'rr_lat'
    # POSITIVE_REGRESSOR_LATENT_BETA_MEAN = 'pr_lat_mean'
    # POSITIVE_REGRESSOR_LATENT_BETA = 'pr_lat'

    REGRESSOR_LATENT_BETA = 'beta_lat'
    REGRESSOR_BETA = 'beta'


# Defaults Values
DEFAULT_LEVEL_SIGMA = 10
DEFAULT_PR_STEP_SCALE = 0.03
DEFAULT_SPAN_LEVEL = 0.1
DEFAULT_SPAN_REGRESSOR = 0.2
DEFAULT_RHO_LEVEL = 0.05
DEFAULT_RHO_REGRESSOR = 0.15
