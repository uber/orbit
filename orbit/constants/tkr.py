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
    _RESPONSE_SD = 'SDY'
    _DEGREE_OF_FREEDOM = 'DOF'
    # ----------  Level  ---------- #
    _NUM_KNOTS_LEVEL = 'N_KNOTS_LEV'
    _LEVEL_KNOT_SCALE = 'LEV_KNOT_SCALE'
    _KERNEL_LEVEL = 'K_LEV'
    # ----------  Regression  ---------- #
    _NUM_KNOTS_COEFFICIENTS  = 'N_KNOTS_COEF'
    _KERNEL_COEFFICIENTS = 'K_COEF'
    _NUM_OF_REGRESSORS = 'N_RR'
    _REGRESSOR_MATRIX = 'RR'
    _REGRESSOR_KNOT_LOC = 'RR_KNOT_LOC'
    _REGRESSOR_KNOT_SCALE = 'RR_KNOT_SCALE'


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
    COEFFICIENTS_KNOT = 'coef_knot'
    COEFFICIENTS = 'coef'


# Defaults Values
DEFAULT_LEVEL_KNOT_SCALE = 10
DEFAULT_SPAN_LEVEL = 0.1
DEFAULT_SPAN_COEFFICIENTS = 0.2
DEFAULT_RHO_LEVEL = 0.05
DEFAULT_RHO_COEFFICIENTS = 0.15
DEFAULT_COEFFICIENTS_LOC = 0
DEFAULT_COEFFICIENTS_SCALE = 1.0
DEFAULT_DEGREE_OF_FREEDOM = 30
