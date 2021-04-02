import numpy as np
from ..constants import dlt as constants


class DLTInitializer(object):
    def __init__(self, s, n_pr, n_nr, n_rr):
        self.s = s
        self.n_pr = n_pr
        self.n_nr = n_nr
        self.n_rr = n_rr

    def __call__(self):
        init_values = dict()
        if self.s > 1:
            init_sea = np.clip(np.random.normal(loc=0, scale=0.05, size=self.s - 1), -1.0, 1.0)
            # catch cases with extreme values
            # init_sea[init_sea > 1.0] = 1.0
            # init_sea[init_sea < -1.0] = -1.0
            init_values[constants.LatentSamplingParameters.INITIAL_SEASONALITY.value] = init_sea
        if self.n_pr > 0:
            x = np.clip(np.random.normal(loc=0, scale=0.1, size=self.n_pr), 0.0, 1.0)
            init_values[constants.LatentSamplingParameters.REGRESSION_POSITIVE_COEFFICIENTS.value] = x
        if self.n_nr > 0:
            x = np.clip(np.random.normal(loc=-0, scale=0.1, size=self.n_nr), -1.0, 0.0)
            init_values[constants.LatentSamplingParameters.REGRESSION_NEGATIVE_COEFFICIENTS.value] = x
        if self.n_rr > 0:
            x = np.random.normal(loc=-0, scale=0.1, size=self.n_rr)
            init_values[constants.LatentSamplingParameters.REGRESSION_REGULAR_COEFFICIENTS.value] = x

        return init_values
