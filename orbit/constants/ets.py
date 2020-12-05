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
    _RESPONSE_SD = 'RESPONSE_SD'
    # ---------- Seasonality ---------- #
    _SEASONALITY = 'SEASONALITY'
    _SEASONALITY_SM_INPUT = 'SEA_SM_INPUT'
    # ---------- Common Local Trend ---------- #
    _LEVEL_SM_INPUT = 'LEV_SM_INPUT'
    _WITH_MCMC = 'WITH_MCMC'


class BaseSamplingParameters(Enum):
    """
    The stan output sampling parameters related with LGT base model.
    """
    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = 'l'
    LEVEL_SMOOTHING_FACTOR = 'lev_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'


class SeasonalitySamplingParameters(Enum):
    """
    The stan output sampling parameters related with seasonality component.
    """
    SEASONALITY_LEVELS = 's'
    SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'
