import numpy as np
from ..constants import ets as constants


class ETSInitializer(object):
    def __init__(self, s):
        self.s = s

    def __call__(self):
        init_values = dict()
        init_sea = np.random.normal(loc=0, scale=0.05, size=self.s - 1)
        # catch cases with extreme values
        init_sea[init_sea > 1.0] = 1.0
        init_sea[init_sea < -1.0] = -1.0
        init_values[constants.LatentSamplingParameters.INITIAL_SEASONALITY.value] = init_sea
        return init_values
