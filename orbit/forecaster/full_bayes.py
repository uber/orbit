import numpy as np
import pandas as pd
from functools import partial

from ..constants.constants import PredictMethod, PredictionKeys
from ..exceptions import ForecasterException
from ..utils.predictions import prepend_date_column, compute_percentiles
from .forecaster import Forecaster


class FullBayesianForecaster(Forecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # method we used to define posteriors and prediction parameters
        self._point_method = None
        # init aggregate posteriors
        self._point_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }

    def set_forecaster_training_meta(self, data_input):
        # MCMC flag to be true
        data_input.update({"WITH_MCMC": 1})
        return data_input

    def fit(
        self,
        df,
        point_method=None,
        keep_samples=True,
        sampling_temperature=1.0,
        **kwargs,
    ):
        super().fit(df, sampling_temperature=sampling_temperature, **kwargs)
        self._point_method = point_method

        if point_method is not None:
            if point_method not in [
                PredictMethod.MEAN.value,
                PredictMethod.MEDIAN.value,
            ]:
                raise ForecasterException("Invalid point estimate method.")

        mean_posteriors = {}
        median_posteriors = {}

        # for each model param, aggregate using `method`
        for param_name in self._model.get_model_param_names():
            param_ndarray = self._posterior_samples[param_name]
            mean_posteriors.update(
                {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
            )
            median_posteriors.update(
                {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
            )

        self._point_posteriors[PredictMethod.MEAN.value] = mean_posteriors
        self._point_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

        if point_method is not None and not keep_samples:
            self._posterior_samples = {}

        self.load_extra_methods()

        return self

    @staticmethod
    def _bootstrap(num_samples, posterior_samples, n):
        """Draw `n` number of bootstrap samples from the posterior_samples.
        Args
        ----
        n : int
            The number of bootstrap samples to draw
        """
        if n < 2:
            raise ForecasterException(
                "Error: Number of bootstrap draws must be at least 2"
            )

        sample_idx = np.random.choice(range(num_samples), size=n, replace=True)
        bootstrap_samples_dict = {}
        for k, v in posterior_samples.items():
            bootstrap_samples_dict[k] = v[sample_idx]

        return bootstrap_samples_dict

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

        if self._point_method is None:
            # full posteriors prediction
            # if bootstrap draws, replace posterior samples with bootstrap
            posterior_samples = (
                self._bootstrap(
                    num_samples=self.estimator.num_sample,
                    posterior_samples=self._posterior_samples,
                    n=self._n_bootstrap_draws,
                )
                if self._n_bootstrap_draws > 1
                else self._posterior_samples
            )

            predicted_dict = self._model.predict(
                posterior_estimates=posterior_samples,
                df=df,
                training_meta=training_meta,
                prediction_meta=prediction_meta,
                include_error=True,
                **kwargs,
            )

            if PredictionKeys.PREDICTION.value not in predicted_dict.keys():
                raise ForecasterException(
                    "cannot find the key:'{}' from return of _predict()".format(
                        PredictionKeys.PREDICTION.value
                    )
                )

            # reduce to prediction only if decompose is not requested
            if not decompose:
                predicted_dict = {
                    k: v
                    for k, v in predicted_dict.items()
                    if k == PredictionKeys.PREDICTION.value
                }

            if store_prediction_array:
                self.prediction_array = predicted_dict[PredictionKeys.PREDICTION.value]
            percentiles_dict = compute_percentiles(
                predicted_dict, self._prediction_percentiles
            )
            predicted_df = pd.DataFrame(percentiles_dict)
            predicted_df = prepend_date_column(predicted_df, df, self.date_col)
            return predicted_df
        else:
            # perform point prediction
            point_posteriors = self._point_posteriors.get(self._point_method)
            point_predicted_dict = self._model.predict(
                posterior_estimates=point_posteriors,
                df=df,
                training_meta=training_meta,
                prediction_meta=prediction_meta,
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
                    posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)
                predicted_dict = self._model.predict(
                    posterior_estimates=posterior_samples,
                    df=df,
                    training_meta=training_meta,
                    prediction_meta=prediction_meta,
                    include_error=True,
                    **kwargs,
                )
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

                # reduce to prediction only if decompose is not requested
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

    def load_extra_methods(self):
        for method in self.extra_methods:
            setattr(
                self,
                method,
                partial(
                    getattr(self._model, method),
                    self.get_training_meta(),
                    self._point_method,
                    self.get_point_posteriors(),
                    self.get_posterior_samples(),
                ),
            )

    def get_wbic(self):
        # This function calculates the WBIC given that MCMC sampling happened with sampling_temperature = log(n)
        training_metrics = self.get_training_metrics()  # get the training metrics
        training_meta = self.get_training_meta()  # get the meta data
        sampling_temp = training_metrics[
            "sampling_temperature"
        ]  # get the sampling temperature
        nobs = training_meta["num_of_obs"]  # the number of observations
        if sampling_temp != np.log(nobs):
            raise ForecasterException(
                "Sampling temperature is not log(n); WBIC calculation is not valid!"
            )
        return -2 * np.nanmean(training_metrics["loglk"]) * nobs

    def fit_wbic(self, df):
        """This function calculates the WBIC for a Orbit model
        Note that if sampling has not been done ith sampling_temperature = log(n) then
        the MCMC sampling is redone to get the WBIC
        """
        nobs = df.shape[0]
        self.fit(df, sampling_temperature=np.log(nobs))
        return self.get_wbic()
