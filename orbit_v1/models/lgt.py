import pandas as pd


class LGTFull(object):
    def __init__(self, estimator_class, model_params, **kwargs):
        # todo: tbd if we pass concrete or abstract estimator class
        self.estimator_class = estimator_class  # abstract
        self.model_params = model_params
        self.estimator = estimator_class(**kwargs)  # concrete

    def _convert_to_estimator_inputs(self):
        pass

    def fit(self, df):
        stan_inputs = self._convert_to_estimator_inputs(df)
        self.estimator.fit(stan_inputs)

    def _predict(self, df):
        posteriors = self.estimator.get_posteriors()
        predict_out = predict_math(posteriors, df)
        return predict_out


class LGTAggregated(object):
    def __init__(self, estimator_class, model_params, **kwargs):
        pass


class LGTMAP(object):
    def __init__(self, estimator_class, model_params, **kwargs):
        pass