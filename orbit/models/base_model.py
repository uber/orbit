from ..estimators.stan_estimator import StanEstimatorMCMC


class BaseModel(object):
    """Base model class

    `BaseModel` will instantiate an estimator class of `estimator_type`.

    Each model defines its own `_supported_estimator_types` to determine if
    the provided `estimator_type` is supported for that particular model.

    Parameters
    ----------
    estimator_type : orbit.BaseEstimator
        Any subclass of `orbit.BaseEstimator`

    """
    # data input mapper to stan model
    _data_input_mapper = None
    # stan model name (e.g. name of `*.stan` file in package)
    _model_name = None
    # supported estimators in ..estimators
    _supported_estimator_types = None  # set for each model

    def __init__(self, estimator_type=StanEstimatorMCMC, **kwargs):
        self.estimator_type = estimator_type

        # create concrete estimator object
        self.estimator = self.estimator_type(**kwargs)
