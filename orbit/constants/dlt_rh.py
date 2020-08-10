from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to stan file
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    TIME_DELTA = 'TIME_DELTA'
    # observation related
    NUM_OF_OBSERVATIONS = 'NUM_OF_OBS'
    RESPONSE = 'RESPONSE'
    INIT_LEV = 'INIT_LEV'
    INIT_LEV_SD = 'INIT_LEV_SD'
    RESPONSE_SD = 'RESPONSE_SD'
    DAMPED_FACTOR = 'DAMPED_FACTOR'
    # ---------- Seasonality ---------- #
    SEASONALITY = 'SEASONALITY'
    NORMALIZE_SEASONALITY = 'NORM_SEAS'
    # ----------  Noise Distribution  ---------- #
    MIN_NU = 'MIN_NU'
    MAX_NU = 'MAX_NU'
    CAUCHY_SD = 'CAUCHY_SD'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with DLT base model.
    """
    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = 'l'
    LOCAL_TREND_SLOPE = 'b'
    LEVEL_SMOOTHING_FACTOR = 'l_sm'
    SLOPE_SMOOTHING_FACTOR = 'b_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'
    RESIDUAL_DEGREE_OF_FREEDOM = 'nu'
    GLOBAL_TREND_SLOPE = 'gb'
    # GLOBAL_TREND_LEVEL = 'gl'
    GLOBAL_TREND = 'gt_sum'


class SeasonalitySamplingParameters(Enum):
    """
    The stan output sampling parameters related with seasonality component.
    """
    SEASONALITY_LEVELS = 's'
    # we don't need initial seasonality
    # INITIAL_SEASONALITY = 'init_sea'
    SEASONALITY_SMOOTHING_FACTOR = 's_sm'
