import numpy as np
from copy import deepcopy
import torch
from enum import Enum

from ..constants.constants import PredictionKeys, TrainingMetaKeys, PredictionMetaKeys
from ..exceptions import IllegalArgument, DataInputException
from .model_template import ModelTemplate
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP
from ..utils.features import moving_average


# constants for attributes, params and I/Os
class DataInputMapper(Enum):
    """
    mapping from object input to sampler
    """

    # ---------- Seasonality ---------- #
    _SEASONALITY = "SEASONALITY"
    SEASONALITY_SD = "SEASONALITY_SD"
    _SEASONALITY_SM_INPUT = "SEA_SM_INPUT"
    # ---------- Common Local Trend ---------- #
    _LEVEL_SM_INPUT = "LEV_SM_INPUT"
    # handle missing values
    IS_VALID_RESPONSE = "IS_VALID_RES"


class BaseSamplingParameters(Enum):
    """
    base parameters in posteriors sampling
    """

    # ---------- Common Local Trend ---------- #
    LOCAL_TREND_LEVELS = "l"
    LEVEL_SMOOTHING_FACTOR = "lev_sm"
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = "obs_sigma"


class SeasonalitySamplingParameters(Enum):
    """
    seasonality component related parameters in posteriors sampling
    """

    SEASONALITY_LEVELS = "s"
    SEASONALITY_SMOOTHING_FACTOR = "sea_sm"


class LatentSamplingParameters(Enum):
    """
    latent variables to be sampled
    """

    INITIAL_SEASONALITY = "init_sea"


class ETSModel(ModelTemplate):
    """
    Parameters
    ----------
    seasonality : int
        Length of seasonality
    seasonality_sm_input : float
        float value between [0, 1], applicable only if `seasonality` > 1. A larger value puts
        more weight on the current seasonality.
        If None, the model will estimate this value.
    level_sm_input : float
        float value between [0.0001, 1]. A larger value puts more weight on the current level.
        If None, the model will estimate this value.
    """

    # data labels for sampler
    _data_input_mapper = DataInputMapper
    # used to match name of `*.stan` or `*.pyro` file to look for the model
    _model_name = "ets"
    _supported_estimator_types = [StanEstimatorMAP, StanEstimatorMCMC]

    def __init__(
        self, seasonality=None, seasonality_sm_input=None, level_sm_input=None, **kwargs
    ):
        # estimator is created in base class
        super().__init__(**kwargs)
        self.seasonality = seasonality
        self.seasonality_sd = None

        # fixed smoothing parameters config
        self.seasonality_sm_input = seasonality_sm_input
        self.level_sm_input = level_sm_input

        # set private var to arg value
        # if None set default in _set_default_args()
        self._seasonality = self.seasonality
        self._seasonality_sm_input = self.seasonality_sm_input
        self._level_sm_input = self.level_sm_input
        # handle missing values
        self.is_valid_response = None

        self._set_static_attributes()
        self._set_model_param_names()

    def _set_static_attributes(self):
        """Set attributes which are independent from data"""
        # setting defaults and proper data type
        if self.seasonality_sm_input is None:
            self._seasonality_sm_input = -1
        if self.level_sm_input is None:
            self._level_sm_input = -1
        elif self.level_sm_input < 0 or self.level_sm_input > 1:
            raise IllegalArgument(
                "only values between [0, 1] are supported for level_sm_input "
                "to build a model with meaningful trend."
            )
        if self.seasonality is None:
            self._seasonality = -1

    def set_dynamic_attributes(self, df, training_meta):
        """Set attributes which are dependent on data"""
        # compute data-driven prior for seasonality
        if self._seasonality > 1:
            response = training_meta[TrainingMetaKeys.RESPONSE.value]
            response_ma = moving_average(
                response, window=self._seasonality, mode="same"
            )
            adjusted_response = response - response_ma
            # to estimate the "across-group" s.d. as a seasonality prior
            ss = np.zeros(self._seasonality)
            for idx in range(self._seasonality):
                ss[idx] = np.nanmean(adjusted_response[idx :: self._seasonality])
            self.seasonality_sd = np.nanstd(ss)
        else:
            # should not be used anyway; just a placeholder
            self.seasonality_sd = training_meta[TrainingMetaKeys.RESPONSE_SD.value]

        self._set_valid_response(training_meta)

    def _set_valid_response(self, training_meta):
        response = training_meta[TrainingMetaKeys.RESPONSE.value]
        self.is_valid_response = (~np.isnan(response)).astype(int)
        # raise exception if the first response value is missing
        if self.is_valid_response[0] == 0:
            raise DataInputException(
                "The first value of response column {} cannot be missing..".format(
                    training_meta[TrainingMetaKeys.RESPONSE_COL.value]
                )
            )

    def set_init_values(self):
        """Override function from base class"""
        init_values = dict()
        if self._seasonality > 1:
            init_sea = np.clip(
                np.random.normal(loc=0, scale=0.05, size=self._seasonality - 1),
                -1.0,
                1.0,
            )
            init_values[LatentSamplingParameters.INITIAL_SEASONALITY.value] = init_sea
            self._init_values = init_values

    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api"""
        self._model_param_names += [param.value for param in BaseSamplingParameters]

        # append seasonality param names
        if self._seasonality > 1:
            self._model_param_names += [
                param.value for param in SeasonalitySamplingParameters
            ]

    def predict(
        self,
        posterior_estimates,
        df,
        training_meta,
        prediction_meta,
        include_error=False,
        **kwargs,
    ):
        """Vectorized version of prediction math"""
        ################################################################
        # Prediction Attributes
        ################################################################
        # n_forecast_steps = prediction_meta[PredictionMetaKeys.FUTURE_STEPS.value]
        start = prediction_meta[PredictionMetaKeys.START_INDEX.value]
        trained_len = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        full_len = prediction_meta[PredictionMetaKeys.END_INDEX.value]

        ################################################################
        # Model Attributes
        ################################################################
        model = deepcopy(posterior_estimates)
        for k, v in model.items():
            model[k] = torch.from_numpy(v)

        # We can pull any arbitrary value from teh dictionary because we hold the
        # safe assumption: the length of the first dimension is always the number of samples
        # thus can be safely used to determine `num_sample`. If predict_method is anything
        # other than full, the value here should be 1
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        # seasonality components
        seasonality_levels = model.get(
            SeasonalitySamplingParameters.SEASONALITY_LEVELS.value
        )
        seasonality_smoothing_factor = model.get(
            SeasonalitySamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend components
        level_smoothing_factor = model.get(
            BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value
        )
        local_trend_levels = model.get(BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        residual_sigma = model.get(BaseSamplingParameters.RESIDUAL_SIGMA.value)

        ################################################################
        # Seasonality Component
        ################################################################

        # calculate seasonality component
        if self._seasonality > 1:
            if full_len <= seasonality_levels.shape[1]:
                seasonal_component = seasonality_levels[:, :full_len]
            else:
                seasonality_forecast_length = full_len - seasonality_levels.shape[1]
                seasonality_forecast_matrix = torch.zeros(
                    (num_sample, seasonality_forecast_length), dtype=torch.double
                )
                seasonal_component = torch.cat(
                    (seasonality_levels, seasonality_forecast_matrix), dim=1
                )
        else:
            seasonal_component = torch.zeros((num_sample, full_len), dtype=torch.double)

        ################################################################
        # Trend Component
        ################################################################

        # calculate level component.
        # However, if predicted end of period > training period, update with out-of-samples forecast
        if full_len <= trained_len:
            trend_component = local_trend_levels[:, :full_len]
            # in-sample error are iids
            if include_error:
                error_value = np.random.normal(
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=trend_component.shape,
                )

                error_value = torch.from_numpy(error_value).double()
                trend_component += error_value
        else:
            trend_component = local_trend_levels
            trend_forecast_length = full_len - trained_len
            trend_forecast_init = torch.zeros(
                (num_sample, trend_forecast_length), dtype=torch.double
            )
            # in-sample error are iids
            if include_error:
                error_value = np.random.normal(
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=trend_component.shape,
                )

                error_value = torch.from_numpy(error_value).double()
                trend_component += error_value

            trend_component = torch.cat((trend_component, trend_forecast_init), dim=1)

            last_local_trend_level = local_trend_levels[:, -1]

            for idx in range(trained_len, full_len):
                trend_component[:, idx] = last_local_trend_level

                if include_error:
                    error_value = np.random.normal(
                        scale=residual_sigma, size=num_sample
                    )
                    error_value = torch.from_numpy(error_value).double()
                    trend_component[:, idx] += error_value

                new_local_trend_level = (
                    level_smoothing_factor * trend_component[:, idx]
                    + (1 - level_smoothing_factor) * last_local_trend_level
                )

                if self._seasonality > 1 and idx + self._seasonality < full_len:
                    seasonal_component[:, idx + self._seasonality] = (
                        seasonality_smoothing_factor.flatten()
                        * (
                            trend_component[:, idx]
                            + seasonal_component[:, idx]
                            - new_local_trend_level
                        )
                        + (1 - seasonality_smoothing_factor.flatten())
                        * seasonal_component[:, idx]
                    )

                last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = trend_component[:, start:]
        seasonal_component = seasonal_component[:, start:]

        # sum components
        pred_array = trend_component + seasonal_component

        pred_array = pred_array.numpy()
        trend_component = trend_component.numpy()
        seasonal_component = seasonal_component.numpy()

        out = {
            PredictionKeys.PREDICTION.value: pred_array,
            PredictionKeys.TREND.value: trend_component,
            PredictionKeys.SEASONALITY.value: seasonal_component,
        }

        return out
