from ..estimators.stan_estimator import StanEstimatorMCMC


class BaseModel(object):
    # data input mapper to stan model
    _data_input_mapper = None
    # stan model name (e.g. name of `*.stan` file in package)
    _stan_model_name = None
    # supported estimators in ..estimators
    _supported_estimator_types = None  # set for each model

    def __init__(self, estimator_type=StanEstimatorMCMC, **kwargs):
        self.estimator_type = estimator_type

        # create concrete estimator object
        self.estimator = self.estimator_type(**kwargs)
