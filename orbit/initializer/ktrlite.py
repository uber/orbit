import numpy as np
from ..constants import ktrlite as constants


class KTRLiteInitializer(object):
    def __init__(self, num_regressor, num_knots_coefficients):
        self.num_regressor = num_regressor
        self.num_knots_coefficients = num_knots_coefficients

    def __call__(self):
        init_values = dict()
        if self.num_regressor> 1:
            init_values[constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value] = np.zeros(
                (self.num_regressor, self.num_knots_coefficients)
            )
        return init_values
