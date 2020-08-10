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
    RESPONSE_SD = 'RESPONSE_SD'
    # ----------  Noise Distribution  ---------- #
    MIN_NU = 'MIN_NU'
    MAX_NU = 'MAX_NU'
    CAUCHY_SD = 'CAUCHY_SD'
    NUM_OF_REGRESSOR = 'NUM_OF_REG'
    REGRESSOR_MATRIX = 'REG_MAT'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with DLT base model.
    """
    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = 'l'
    LEVEL_SMOOTHING_FACTOR = 'l_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'
    RESIDUAL_DEGREE_OF_FREEDOM = 'nu'
    GLOBAL_TREND_SLOPE = 'gb'
    GLOBAL_TREND = 'gt_sum'
    YHAT = 'yhat'


class RegressionParameters(Enum):
    """
    """
    REGRESSION_SMOOTHING_FACTOR = 'b_sm'
    REGRESSION_COEF = 'b'
    REG = 'reg'