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
    DEGREE_OF_FREEDOM = 'DOF'
    MIN_RESIDUALS_SD = 'MIN_RESIDUALS_SD'
    # ----------  Level  ---------- #
    _NUM_KNOTS_LEVEL = 'N_KNOTS_LEV'
    LEVEL_KNOT_SCALE = 'LEV_KNOT_SCALE'
    _KERNEL_LEVEL = 'K_LEV'
    # ----------  Regression  ---------- #
    _NUM_KNOTS_COEFFICIENTS  = 'N_KNOTS_COEF'
    _KERNEL_COEFFICIENTS = 'K_COEF'
    _NUM_OF_REGULAR_REGRESSORS = 'N_RR'
    _NUM_OF_POSITIVE_REGRESSORS = 'N_PR'
    _REGULAR_REGRESSOR_MATRIX = 'RR'
    _POSITIVE_REGRESSOR_MATRIX = 'PR'
    _REGULAR_REGRESSOR_INIT_KNOT_LOC = 'RR_INIT_KNOT_LOC'
    _REGULAR_REGRESSOR_INIT_KNOT_SCALE = 'RR_INIT_KNOT_SCALE'
    _REGULAR_REGRESSOR_KNOT_SCALE = 'RR_KNOT_SCALE'
    _POSITIVE_REGRESSOR_INIT_KNOT_LOC = 'PR_INIT_KNOT_LOC'
    _POSITIVE_REGRESSOR_INIT_KNOT_SCALE = 'PR_INIT_KNOT_SCALE'
    _POSITIVE_REGRESSOR_KNOT_SCALE = 'PR_KNOT_SCALE'
    # ----------  Prior Specification  ---------- #
    _COEF_PRIOR_LIST = 'COEF_PRIOR_LIST'
    _LEVEL_KNOTS = 'LEV_KNOT_LOC'
    _SEAS_TERM = 'SEAS_TERM'
    # --------------- mvn
    MVN = 'MVN'
    GEOMETRIC_WALK = 'GEOMETRIC_WALK'


class BaseSamplingParameters(Enum):
    """
    The output sampling parameters related with the base model
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
    COEFFICIENTS_INIT_KNOT= 'coef_init_knot'
    COEFFICIENTS = 'coef'


# Defaults Values
DEFAULT_REGRESSOR_SIGN = '='
DEFAULT_COEFFICIENTS_INIT_KNOT_SCALE = 1.0
DEFAULT_COEFFICIENTS_INIT_KNOT_LOC = 0
DEFAULT_COEFFICIENTS_KNOT_SCALE = 0.1
DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER = 0.01
DEFAULT_UPPER_BOUND_SCALE_MULTIPLIER = 1.0
