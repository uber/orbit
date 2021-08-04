import numpy as np
import pandas as pd

from ..constants.constants import PredictMethod, PredictionKeys
from ..exceptions import ForecasterException
from ..utils.predictions import prepend_date_column, compute_percentiles
from .forecaster import Forecaster


class SVIForecaster(Forecaster):
    def __init__(self,
                 estimate_method=None,
                 **kwargs):
        super().__init__(**kwargs)
        # extra fields for svi and mcmc methods
        self.estimate_method = estimate_method
        self._n_bootstrap_draws = self.n_bootstrap_draws
        if not self.n_bootstrap_draws:
            self._n_bootstrap_draws = -1

        # init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }

    @staticmethod
    def _bootstrap(num_samples, posterior_samples, n):
        """Draw `n` number of bootstrap samples from the posterior_samples.
        Args
        ----
        n : int
            The number of bootstrap samples to draw
        """
        if n < 2:
            raise ForecasterException("Error: Number of bootstrap draws must be at least 2")

        sample_idx = np.random.choice(range(num_samples), size=n, replace=True)
        bootstrap_samples_dict = {}
        for k, v in posterior_samples.items():
            bootstrap_samples_dict[k] = v[sample_idx]

        return bootstrap_samples_dict

    # def fit(self, df):
    #
    #
    #     self._posterior_samples = _posterior_samples
    #     mean_posteriors = {}
    #     median_posteriors = {}
    #
    #     # for each model param, aggregate using `method`
    #     for param_name in self._model.get_model_param_names():
    #         param_ndarray = _posterior_samples[param_name]
    #         mean_posteriors.update(
    #             {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
    #         )
    #         median_posteriors.update(
    #             {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
    #         )
    #
    #     self._aggregate_posteriors[PredictMethod.MEAN.value] = mean_posteriors
    #     self._aggregate_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

    def predict(self, df, decompose=False, store_prediction_array=False, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise ForecasterException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self._set_prediction_meta(df)
        prediction_meta = self.get_prediction_meta()
        training_meta = self.get_training_meta()

        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(
            num_samples=self.estimator.num_sample,
            posterior_samples=self._posterior_samples,
            n=self._n_bootstrap_draws
        ) if self._n_bootstrap_draws > 1 else self._posterior_samples

        predicted_dict = self._model.predict(
            posterior_estimates=posterior_samples,
            df=df,
            training_meta=training_meta,
            prediction_meta=prediction_meta,
            include_error=True,
            **kwargs
        )

        if PredictionKeys.PREDICTION.value not in predicted_dict.keys():
            raise ForecasterException("cannot find the key:'{}' from return of _predict()".format(
                PredictionKeys.PREDICTION.value))

        # reduce to prediction only if decompose is not requested
        if not decompose:
            predicted_dict = {k: v for k, v in predicted_dict.items() if k == PredictionKeys.PREDICTION.value}

        if store_prediction_array:
            self.prediction_array = predicted_dict[PredictionKeys.PREDICTION.value]
        percentiles_dict = compute_percentiles(predicted_dict, self._prediction_percentiles)
        predicted_df = pd.DataFrame(percentiles_dict)
        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df
