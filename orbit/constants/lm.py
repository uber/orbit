from enum import Enum


class StanDataInput(Enum):
    NUM_OF_OBSERVATIONS = 'N'
    NUM_OF_REGRESSORS = 'P'
    RESPONSE = 'y'
    REGRESSOR_MATRIX = 'X'

class StanSampleOutput(Enum):
    INTERCEPT = 'alpha'
    COEFFICIENTS = 'beta'
    OBS_ERROR_SCALE = 'sigma'
