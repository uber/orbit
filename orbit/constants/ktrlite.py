from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to pyro input
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    _NUM_OF_OBSERVATIONS = 'N_OBS'
    _RESPONSE = 'RESPONSE'
    # used for pyro
    # _IS_VALID_RESPONSE = 'IS_VALID_RES'
    _NUM_OF_VALID_RESPONSE = 'N_VALID_RES'
    # mainly used for stan
    _WHICH_VALID_RESPONSE = 'WHICH_VALID_RES'
    _RESPONSE_SD = 'SDY'
    _RESPONSE_MEAN = 'MEAN_Y'
    _DEGREE_OF_FREEDOM = 'DOF'
    # ----------  Level  ---------- #
    _NUM_KNOTS_LEVEL = 'N_KNOTS_LEV'
    LEVEL_KNOT_SCALE = 'LEV_KNOT_SCALE'
    _KERNEL_LEVEL = 'K_LEV'
    # ----------  Regression  ---------- #
    _NUM_KNOTS_COEFFICIENTS  = 'N_KNOTS_COEF'
    _KERNEL_COEFFICIENTS = 'K_COEF'
    _NUM_OF_REGRESSORS = 'P'
    _REGRESSOR_MATRIX = 'REGRESSORS'
    _COEFFICIENTS_KNOT_POOLING_LOC = 'COEF_KNOT_POOL_LOC'
    _COEFFICIENTS_KNOT_POOLING_SCALE = 'COEF_KNOT_POOL_SCALE'
    _COEFFICIENTS_KNOT_SCALE = 'COEF_KNOT_SCALE'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with LGT base model.
    """
    LEVEL_KNOT = 'lev_knot'
    LEVEL = 'lev'
    YHAT = 'yhat'
    OBS_SCALE = 'obs_scale'


class RegressionSamplingParameters(Enum):
    """
    The stan output sampling parameters related with regression component.
    """

    # REGULAR_REGRESSOR_LATENT_BETA = 'rr_lat'
    # POSITIVE_REGRESSOR_LATENT_BETA_MEAN = 'pr_lat_mean'
    # POSITIVE_REGRESSOR_LATENT_BETA = 'pr_lat'

    COEFFICIENTS_KNOT = 'coef_knot'
    COEFFICIENTS = 'coef'
