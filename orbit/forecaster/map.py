import numpy as np
import pandas as pd

from ..constants.constants import PredictMethod, PredictionKeys
from ..exceptions import ForecasterException
from ..utils.predictions import prepend_date_column, compute_percentiles
from .forecaster import Forecaster


class MAPForecaster(Forecaster):
    def __init__(self, n_bootstrap_draws=1e4, **kwargs):
        super().__init__(n_bootstrap_draws=n_bootstrap_draws, **kwargs)

    def is_fitted(self):
        # if either aggregate posterior and posterior_samples are non-empty, claim it as fitted model (true),
        # else false.
        return bool(self._point_posteriors['map'])

    def fit(self, df):
        super().fit(df)
        # FIXME: shouldn't the expand dim be done inside estimator to make this layer consistent?
        posterior_samples = self._posterior_samples
        map_posterior = {}
        for param_name in self._model.get_model_param_names():
            param_array = posterior_samples[param_name]
            # add dimension so it works with vector math in `_predict`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update({param_name: param_array})

        self._point_posteriors[PredictMethod.MAP.value] = map_posterior

    def predict(self, df, decompose=False, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise ForecasterException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self._set_prediction_meta(df)
        prediction_meta = self.get_prediction_meta()
        training_meta = self.get_training_meta()

        # perform point prediction
        point_posteriors = self._point_posteriors.get(PredictMethod.MAP.value)
        point_predicted_dict = self._model.predict(
            posterior_estimates=point_posteriors,
            df=df,
            training_meta=training_meta,
            prediction_meta=prediction_meta,
            # false for point estimate
            include_error=False,
            **kwargs
        )
        for k, v in point_predicted_dict.items():
            point_predicted_dict[k] = np.squeeze(v, 0)

        # to derive confidence interval; the condition should be sufficient since we add [50] by default
        if self.n_bootstrap_draws > 0 and len(self._prediction_percentiles) > 1:
            # perform bootstrap; we don't have posterior samples. hence, we just repeat the draw here.
            posterior_samples = {}
            for k, v in point_posteriors.items():
                posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)
            predicted_dict = self._model.predict(
                posterior_estimates=posterior_samples,
                df=df,
                training_meta=training_meta,
                prediction_meta=prediction_meta,
                include_error=True,
                **kwargs
            )
            percentiles_dict = compute_percentiles(predicted_dict, self._prediction_percentiles)
            # replace mid point prediction by point estimate
            percentiles_dict.update(point_predicted_dict)

            if PredictionKeys.PREDICTION.value not in percentiles_dict.keys():
                raise ForecasterException("cannot find the key:'{}' from return of _predict()".format(
                    PredictionKeys.PREDICTION.value))

            # since we always assume to have decompose from .predict() at first,
            # here it reduces to prediction when decompose is not requested
            if not decompose:
                k = PredictionKeys.PREDICTION.value
                reduced_keys = [k + "_" + str(p) if p != 50 else k for p in self._prediction_percentiles]
                percentiles_dict = {k: v for k, v in percentiles_dict.items() if k in reduced_keys}
            predicted_df = pd.DataFrame(percentiles_dict)
        else:
            predicted_df = pd.DataFrame(point_predicted_dict)

        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df
