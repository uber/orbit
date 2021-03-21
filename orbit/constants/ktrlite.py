from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to pyro input
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    NUM_OF_OBSERVATIONS = 'N_OBS'
    RESPONSE = 'RESPONSE'
    NUM_OF_VALID_RESPONSE = 'N_VALID_RES'
    WHICH_VALID_RESPONSE = 'WHICH_VALID_RES'
    RESPONSE_SD = 'SDY'
    RESPONSE_OFFSET = 'MEAN_Y'
    _DEGREE_OF_FREEDOM = 'DOF'
    # ----------  Level  ---------- #
    NUM_KNOTS_LEVEL = 'N_KNOTS_LEV'
    LEVEL_KNOT_SCALE = 'LEV_KNOT_SCALE'
    KERNEL_LEVEL = 'K_LEV'
    # ----------  Regression  ---------- #
    NUM_KNOTS_COEFFICIENTS  = 'N_KNOTS_COEF'
    KERNEL_COEFFICIENTS = 'K_COEF'
    NUM_OF_REGRESSORS = 'P'
    REGRESSOR_MATRIX = 'REGRESSORS'
    COEFFICIENTS_KNOT_POOLING_LOC = 'COEF_KNOT_POOL_LOC'
    COEFFICIENTS_KNOT_POOLING_SCALE = 'COEF_KNOT_POOL_SCALE'
    COEFFICIENTS_KNOT_SCALE = 'COEF_KNOT_SCALE'


class BaseSamplingParameters(Enum):
    """
    The output sampling parameters related with base model.
    """
    LEVEL_KNOT = 'lev_knot'
    LEVEL = 'lev'
    YHAT = 'yhat'
    OBS_SCALE = 'obs_scale'


class RegressionSamplingParameters(Enum):
    """
    The output sampling parameters related with regression component.
    """
    COEFFICIENTS_KNOT = 'coef_knot'
    COEFFICIENTS = 'coef'
