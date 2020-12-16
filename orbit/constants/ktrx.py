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
    _NUM_OF_VALID_RESPONSE = 'N_VALID_RES'
    # mainly used for stan
    _WHICH_VALID_RESPONSE = 'WHICH_VALID_RES'
    _RESPONSE_SD = 'SDY'
    _RESPONSE_MEAN = 'MEAN_Y'
    DEGREE_OF_FREEDOM = 'DOF'
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
    _REGULAR_REGRESSOR_KNOT_POOLING_LOC = 'RR_KNOT_POOL_LOC'
    _REGULAR_REGRESSOR_KNOT_POOLING_SCALE = 'RR_KNOT_POOL_SCALE'
    _REGULAR_REGRESSOR_KNOT_SCALE = 'RR_KNOT_SCALE'
    _POSITIVE_REGRESSOR_KNOT_POOLING_LOC = 'PR_KNOT_POOL_LOC'
    _POSITIVE_REGRESSOR_KNOT_POOLING_SCALE = 'PR_KNOT_POOL_SCALE'
    _POSITIVE_REGRESSOR_KNOT_SCALE = 'PR_KNOT_SCALE'
    # ----------  Prior Specification  ---------- #
    # _NUM_INSERT_PRIOR = 'N_PRIOR'
    # _INSERT_PRIOR_MEAN = 'PRIOR_MEAN'
    # _INSERT_PRIOR_SD = 'PRIOR_SD'
    # _INSERT_PRIOR_TP_IDX = 'PRIOR_TP_IDX'
    # _INSERT_PRIOR_IDX = 'PRIOR_IDX'
    _COEF_PRIOR_LIST = 'COEF_PRIOR_LIST'
    _LEVEL_KNOTS = 'LEV_KNOT_LOC'
    _SEAS_TERM = 'SEAS_TERM'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with LGT base model.
    """
    LEVEL_KNOT = 'lev_knot'
    LEVEL = 'lev'
    YHAT = 'yhat'
    OBS_SCALE = 'obs_scale'

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
    COEFFICIENTS_KNOT = 'coef_knot'
    COEFFICIENTS_KNOT_LOCATION = 'coef_knot_loc'
    COEFFICIENTS = 'coef'


# Defaults Values
DEFAULT_REGRESSOR_SIGN = '='
DEFAULT_COEFFICIENTS_KNOT_POOL_SCALE = 1.0
DEFAULT_COEFFICIENTS_KNOT_POOL_LOC = 0
DEFAULT_COEFFICIENTS_KNOT_SCALE = 0.1
