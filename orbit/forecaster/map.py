import numpy as np
import pandas as pd
from functools import partial

from ..constants.constants import PredictMethod, PredictionKeys
from ..exceptions import ForecasterException
from ..utils.predictions import prepend_date_column, compute_percentiles
from .forecaster import Forecaster


class MAPForecaster(Forecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._point_posteriors[PredictMethod.MAP.value] = dict()
        self._point_method = PredictMethod.MAP.value

    def set_forecaster_training_meta(self, data_input):
        # MCMC flag to be true
        data_input.update({"WITH_MCMC": 0})
        return data_input

    def fit(self, df, **kwargs):
        super().fit(df, **kwargs)
        posterior_samples = self._posterior_samples
        map_posterior = {}
        for param_name in self._model.get_model_param_names():
            param_array = posterior_samples[param_name]
            # add one dimension as batch to have consistent logic with `.predict()`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update({param_name: param_array})

        self._point_posteriors[PredictMethod.MAP.value] = map_posterior
        # TODO: right now this is hacky:
        #  need to do it one more time to over-write the extra methods with right posterior
        self.load_extra_methods()

        return self

    def predict(
        self, df, decompose=False, store_prediction_array=False, seed=None, **kwargs
    ) -> pd.DataFrame:
        # raise if model is not fitted
        if not self.is_fitted():
            raise ForecasterException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self._set_prediction_meta(df)
        prediction_meta = self.get_prediction_meta()
        training_meta = self.get_training_meta()

        if seed is not None:
            np.random.seed(seed)

        # perform point prediction
        point_posteriors = self._point_posteriors.get(PredictMethod.MAP.value)
        point_predicted_dict = self._model.predict(
            posterior_estimates=point_posteriors,
            df=df,
            training_meta=training_meta,
            prediction_meta=prediction_meta,
            # false for point estimate
            include_error=False,
            **kwargs,
        )
        for k, v in point_predicted_dict.items():
            point_predicted_dict[k] = np.squeeze(v, 0)

        # to derive confidence interval; the condition should be sufficient since we add [50] by default
        if self._n_bootstrap_draws > 0 and len(self._prediction_percentiles) > 1:
            # perform bootstrap; we don't have posterior samples. hence, we just repeat the draw here.
            posterior_samples = {}
            for k, v in point_posteriors.items():
                posterior_samples[k] = np.repeat(v, self._n_bootstrap_draws, axis=0)
            predicted_dict = self._model.predict(
                posterior_estimates=posterior_samples,
                df=df,
                training_meta=training_meta,
                prediction_meta=prediction_meta,
                include_error=True,
                **kwargs,
            )
            if store_prediction_array:
                self.prediction_array = predicted_dict[PredictionKeys.PREDICTION.value]
            percentiles_dict = compute_percentiles(
                predicted_dict, self._prediction_percentiles
            )
            # replace mid point prediction by point estimate
            percentiles_dict.update(point_predicted_dict)

            if PredictionKeys.PREDICTION.value not in percentiles_dict.keys():
                raise ForecasterException(
                    "cannot find the key:'{}' from return of _predict()".format(
                        PredictionKeys.PREDICTION.value
                    )
                )

            # since we always assume to have decompose from .predict() at first,
            # here it reduces to prediction when decompose is not requested
            if not decompose:
                k = PredictionKeys.PREDICTION.value
                reduced_keys = [
                    k + "_" + str(p) if p != 50 else k
                    for p in self._prediction_percentiles
                ]
                percentiles_dict = {
                    k: v for k, v in percentiles_dict.items() if k in reduced_keys
                }
            predicted_df = pd.DataFrame(percentiles_dict)
        else:
            if not decompose:
                # reduce to prediction only if decompose is not requested
                point_predicted_dict = {
                    k: v
                    for k, v in point_predicted_dict.items()
                    if k == PredictionKeys.PREDICTION.value
                }
            predicted_df = pd.DataFrame(point_predicted_dict)

        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df

    # TODO: should be private
    def load_extra_methods(self):
        for method in self.extra_methods:
            setattr(
                self,
                method,
                partial(
                    getattr(self._model, method),
                    self.get_training_meta(),
                    PredictMethod.MAP.value,
                    self.get_point_posteriors(),
                    self.get_posterior_samples(),
                ),
            )

    def get_bic(self):
        training_metrics = self.get_training_metrics()
        loglk = training_metrics["loglk"]
        n = loglk.shape[0] * loglk.shape[1]
        k = training_metrics["num_of_params"]
        return -2.0 * np.sum(loglk) + k * np.log(n)
