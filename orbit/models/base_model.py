from ..estimators.stan_estimator import StanEstimatorMCMC
from ..utils.docstring_style import merge_numpy_docs_dedup
import custom_inherit as ci
ci.store["numpy_with_merge_dedup"] = merge_numpy_docs_dedup
ci.add_style("numpy_with_merge_dedup", merge_numpy_docs_dedup)


class BaseModel(object, metaclass=ci.DocInheritMeta(style="numpy_with_merge_dedup")):
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
