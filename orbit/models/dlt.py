import pandas as pd
from scipy.stats import nct
import torch
import numpy as np
from copy import deepcopy

from ..constants import dlt as constants
from ..exceptions import IllegalArgument, PredictionException
from ..models.lgt import BaseLGT, LGTFull, LGTAggregated, LGTMAP
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP

from ..utils.general import is_ordered_datetime


class BaseDLT(BaseLGT):
    """Base DLT model object with shared functionality for Full, Aggregated, and MAP methods

    The model arguments are the same as `BaseLGT` with some additional arguments

    Parameters
    ----------
    period : int
        Used to set `time_delta` as `1 / max(period, seasonality)`. If None and no seasonality,
        then `time_delta` == 1
    damped_factor : float
        Hyperparameter float value between [0, 1]. A smaller value further dampens the previous
        global trend value. Default, 0.8
    global_trend_option : { 'flat', 'linear', 'loglinear', 'logistic' }
        Transformation function for the shape of the forecasted global trend.

    See Also
    --------
    orbit.models.lgt.BaseLGT

    """
    _data_input_mapper = constants.DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'dlt'
    _supported_estimator_types = None  # set for each model

    def __init__(self, period=1, damped_factor=0.8, global_trend_option='linear', **kwargs):
        self.damped_factor = damped_factor
        self.global_trend_option = global_trend_option
        self.period = period

        # global trend related attributes
        self._global_trend_option = None
        self._time_delta = 1

        # order matters and super constructor called after attributes are set
        # since we override _set_static_data_attributes()
        super().__init__(**kwargs)

    def _set_global_trend_attributes(self):
        self._global_trend_option = getattr(constants.GlobalTrendOption, self.global_trend_option).value
        self._time_delta = 1 / max(self.period, self._seasonality, 1)

    def _set_static_data_attributes(self):
        super()._set_static_data_attributes()
        self._set_global_trend_attributes()

    def _set_model_param_names(self):
        """Model parameters to extract from Stan"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]

        # append seasonality param names
        if self._seasonality > 1:
            self._model_param_names += [param.value for param in constants.SeasonalitySamplingParameters]

        # append positive regressors if any
        if self._num_of_positive_regressors > 0:
            self._model_param_names += [
                constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value]

        # append regular regressors if any
        if self._num_of_regular_regressors > 0:
            self._model_param_names += [
                constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value]

        if self._global_trend_option != constants.GlobalTrendOption.flat.value:
            self._model_param_names += [param.value for param in constants.GlobalTrendSamplingParameters]

    def _predict(self, posterior_estimates, df=None, include_error=False, decompose=False):

        ################################################################
        # Model Attributes
        ################################################################

        # get model attributes
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
            constants.SeasonalitySamplingParameters.SEASONALITY_LEVELS.value)
        seasonality_smoothing_factor = model.get(
            constants.SeasonalitySamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend components
        slope_smoothing_factor = model.get(
            constants.BaseSamplingParameters.SLOPE_SMOOTHING_FACTOR.value)
        level_smoothing_factor = model.get(
            constants.BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        local_trend_levels = model.get(constants.BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        local_trend_slopes = model.get(constants.BaseSamplingParameters.LOCAL_TREND_SLOPES.value)
        local_trend = model.get(constants.BaseSamplingParameters.LOCAL_TREND.value)
        residual_degree_of_freedom = model.get(
            constants.BaseSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM.value)
        residual_sigma = model.get(constants.BaseSamplingParameters.RESIDUAL_SIGMA.value)

        # set an additional attribute for damped factor when it is fixed
        # get it through user input field
        damped_factor = torch.empty(num_sample, dtype=torch.double)
        damped_factor.fill_(self.damped_factor)

        if self._global_trend_option != constants.GlobalTrendOption.flat.value:
            global_trend_level = model.get(constants.GlobalTrendSamplingParameters.GLOBAL_TREND_LEVEL.value).view(
                num_sample, )
            global_trend_slope = model.get(constants.GlobalTrendSamplingParameters.GLOBAL_TREND_SLOPE.value).view(
                num_sample, )
            global_trend = model.get(constants.GlobalTrendSamplingParameters.GLOBAL_TREND.value)
        else:
            global_trend = torch.zeros(local_trend.shape, dtype=torch.double)

        # regression components
        pr_beta = model.get(constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value)
        rr_beta = model.get(constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value)
        regressor_beta = self._concat_regression_coefs(pr_beta, rr_beta)

        ################################################################
        # Prediction Attributes
        ################################################################

        # get training df meta
        training_df_meta = self._training_df_meta
        # remove reference from original input
        df = df.copy()
        # get prediction df meta
        prediction_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'prediction_start': df[self.date_col].iloc[0],
            'prediction_end': df[self.date_col].iloc[-1]
        }

        if not is_ordered_datetime(prediction_df_meta['date_array']):
            raise IllegalArgument('Datetime index must be ordered and not repeat')

        # TODO: validate that all regressor columns are present, if any

        if prediction_df_meta['prediction_start'] < training_df_meta['training_start']:
            raise PredictionException('Prediction start must be after training start.')

        trained_len = training_df_meta['df_length']
        output_len = prediction_df_meta['df_length']

        # If we cannot find a match of prediction range, assume prediction starts right after train
        # end
        if prediction_df_meta['prediction_start'] > training_df_meta['training_end']:
            forecast_dates = set(prediction_df_meta['date_array'])
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = trained_len
        else:
            # compute how many steps to forecast
            forecast_dates = \
                set(prediction_df_meta['date_array']) - set(training_df_meta['date_array'])
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or \
                               - (len(set(training_df_meta['date_array']) - set(
                                   prediction_df_meta['date_array'])))
            # time index for prediction start
            start = pd.Index(
                training_df_meta['date_array']).get_loc(prediction_df_meta['prediction_start'])

        full_len = trained_len + n_forecast_steps

        ################################################################
        # Regression Component
        ################################################################

        # calculate regression component
        if self.regressor_col is not None and len(self._regular_regressor_col) > 0:
            regressor_beta = regressor_beta.t()
            regressor_matrix = df[self.regressor_col].values
            regressor_torch = torch.from_numpy(regressor_matrix).double()
            regressor_component = torch.matmul(regressor_torch, regressor_beta)
            regressor_component = regressor_component.t()
        else:
            # regressor is always dependent with df. hence, no need to make full size
            regressor_component = torch.zeros((num_sample, output_len), dtype=torch.double)

        ################################################################
        # Seasonality Component
        ################################################################

        # calculate seasonality component
        if self._seasonality > 1:
            if full_len <= seasonality_levels.shape[1]:
                seasonality_component = seasonality_levels[:, :full_len]
            else:
                seasonality_forecast_length = full_len - seasonality_levels.shape[1]
                seasonality_forecast_matrix = \
                    torch.zeros((num_sample, seasonality_forecast_length), dtype=torch.double)
                seasonality_component = torch.cat(
                    (seasonality_levels, seasonality_forecast_matrix), dim=1)
        else:
            seasonality_component = torch.zeros((num_sample, full_len), dtype=torch.double)

        ################################################################
        # Trend Component
        ################################################################

        # calculate level component.
        # However, if predicted end of period > training period, update with out-of-samples forecast
        if full_len <= trained_len:
            full_local_trend = local_trend[:, :full_len]
            full_global_trend = global_trend[:, :full_len]

            # in-sample error are iids
            if include_error:
                error_value = nct.rvs(
                    df=residual_degree_of_freedom.unsqueeze(-1),
                    nc=0,
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=(num_sample, full_len)
                )

                error_value = torch.from_numpy(error_value.reshape(num_sample, full_len)).double()
                full_local_trend += error_value
        else:
            trend_forecast_length = full_len - trained_len
            trend_forecast_init = torch.zeros((num_sample, trend_forecast_length), dtype=torch.double)
            full_local_trend = local_trend[:, :full_len]
            # for convenience, we lump error on local trend since the formula would
            # yield the same as yhat + noise - global_trend - seasonality - regression
            # equivalent with local_trend + noise
            # in-sample error are iids
            if include_error:
                error_value = nct.rvs(
                    df=residual_degree_of_freedom.unsqueeze(-1),
                    nc=0,
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=(num_sample, full_local_trend.shape[1])
                )

                error_value = torch.from_numpy(
                    error_value.reshape(num_sample, full_local_trend.shape[1])).double()
                full_local_trend += error_value

            full_local_trend = torch.cat((full_local_trend, trend_forecast_init), dim=1)
            full_global_trend = torch.cat((global_trend[:, :full_len], trend_forecast_init), dim=1)
            last_local_trend_level = local_trend_levels[:, -1]
            last_local_trend_slope = local_trend_slopes[:, -1]

            for idx in range(trained_len, full_len):
                # based on model, split cases for trend update
                curr_local_trend = \
                    last_local_trend_level + damped_factor.flatten() * last_local_trend_slope
                full_local_trend[:, idx] = curr_local_trend
                # idx = time - 1
                if self.global_trend_option == constants.GlobalTrendOption.linear.name:
                    full_global_trend[:, idx] = \
                        global_trend_level + global_trend_slope * idx * self._time_delta
                elif self.global_trend_option == constants.GlobalTrendOption.loglinear.name:
                    full_global_trend[:, idx] = \
                        global_trend_level + torch.log(1 + global_trend_slope * idx * self._time_delta)
                elif self.global_trend_option == constants.GlobalTrendOption.logistic.name:
                    full_global_trend[:, idx] = \
                        global_trend_level / (1 + torch.exp(-1 * global_trend_slope * idx * self._time_delta))
                elif self._global_trend_option == constants.GlobalTrendOption.flat.name:
                    full_global_trend[:, idx] = 0

                if include_error:
                    error_value = nct.rvs(
                        df=residual_degree_of_freedom,
                        nc=0,
                        loc=0,
                        scale=residual_sigma,
                        size=num_sample
                    )
                    error_value = torch.from_numpy(error_value).double()
                    full_local_trend[:, idx] += error_value

                # now full_local_trend contains the error term and hence we need to use
                # curr_local_trend as a proxy of previous level index
                new_local_trend_level = \
                    level_smoothing_factor * full_local_trend[:, idx] \
                    + (1 - level_smoothing_factor) * curr_local_trend
                last_local_trend_slope = \
                    slope_smoothing_factor * (new_local_trend_level - last_local_trend_level) \
                    + (1 - slope_smoothing_factor) * damped_factor.flatten() * last_local_trend_slope

                if self._seasonality > 1 and idx + self._seasonality < full_len:
                    seasonality_component[:, idx + self._seasonality] = \
                        seasonality_smoothing_factor.flatten() \
                        * (full_local_trend[:, idx] + seasonality_component[:, idx] -
                           new_local_trend_level) \
                        + (1 - seasonality_smoothing_factor.flatten()) * seasonality_component[:,
                                                                         idx]

                last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = full_global_trend[:, start:] + full_local_trend[:, start:]
        seasonality_component = seasonality_component[:, start:]

        # sum components
        pred_array = trend_component + seasonality_component + regressor_component

        pred_array = pred_array.numpy()
        trend_component = trend_component.numpy()
        seasonality_component = seasonality_component.numpy()
        regressor_component = regressor_component.numpy()

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend_component,
                'seasonality': seasonality_component,
                'regression': regressor_component
            }

            return decomp_dict

        return {'prediction': pred_array}


class DLTFull(LGTFull, BaseDLT):
    """Concrete DLT model for full prediction

    The model arguments are the same as `LGTFull` with some additional arguments

    See Also
    --------
    orbit.models.lgt.LGTFull

    """
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DLTAggregated(LGTAggregated, BaseDLT):
    """Concrete DLT model for aggregated posterior prediction

    The model arguments are the same as `LGTAggregated` with some additional arguments

    See Also
    --------
    orbit.models.lgt.LGTAggregated

    """
    _supported_estimator_types = [StanEstimatorMCMC, StanEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DLTMAP(LGTMAP, BaseDLT):
    """Concrete DLT model for MAP (Maximum a Posteriori) prediction

    The model arguments are the same as `LGTNAP` with some additional arguments

    See Also
    --------
    orbit.models.lgt.LGTNAP

    """
    _supported_estimator_types = [StanEstimatorMAP]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
