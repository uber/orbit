from ..exceptions import AbstractMethodException


class ModelTemplate(object):
    """
    Notes
    -----
    contain data structure ; specify what need to fill from abstract to turn a model concrete
    """

    # class attributes
    _data_input_mapper = None
    _model_name = None
    _fitter = None
    _supported_estimator_types = None

    def __init__(self, **kwargs):
        # set by ._set_init_values
        # this is ONLY used by stan which by default used 'random'
        self._init_values = None

        # set by _set_model_param_names()
        self._model_param_names = list()

    def predict(
        self,
        posterior_estimates,
        df,
        training_meta,
        prediction_meta,
        include_error=False,
        **kwargs,
    ):
        """Predict interface for users"""
        raise AbstractMethodException(
            "Abstract method.  Model should implement concrete .predict()."
        )

    def set_dynamic_attributes(self, df, training_meta):
        """Optional; set dynamic fitting input based on input DataFrame, rather than at object instantiation"""
        pass

    def set_init_values(self):
        """Optional; set init as a callable (for Stan ONLY)"""
        pass

    def get_init_values(self):
        return self._init_values

    def get_model_param_names(self):
        return self._model_param_names

    def get_data_input_mapper(self):
        return self._data_input_mapper

    def get_model_name(self):
        return self._model_name

    def get_fitter(self):
        return self._fitter

    def get_supported_estimator_types(self):
        return self._supported_estimator_types
