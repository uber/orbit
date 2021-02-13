from enum import Enum


class DataInputMapper(Enum):
    """
    mapping from object input to sampler
    """
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
    base parameters in posteriors sampling
    """
    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = 'l'
    LEVEL_SMOOTHING_FACTOR = 'lev_sm'
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'


class SeasonalitySamplingParameters(Enum):
    """
    seasonality component related parameters in posteriors sampling
    """
    SEASONALITY_LEVELS = 's'
    SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'


class LatentSamplingParameters(Enum):
    """
    latent variables to be sampled
    """
    INITIAL_SEASONALITY = 'init_sea'
