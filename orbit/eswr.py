import numpy as np
import pandas as pd
from scipy.stats import nct
import torch
from copy import deepcopy
from orbit.estimator import Estimator
from orbit.constants import eswr
from orbit.constants.constants import (
    PredictMethod
)
from orbit.exceptions import (
    PredictionException,
    IllegalArgument
)
from orbit.utils.utils import is_ordered_datetime


class ESWR(Estimator):
    # this must be defined in child class
    _data_input_mapper = eswr.DataInputMapper

    def __init__(
            self,
            is_multiplicative=True,
            regressor_col=None,
            **kwargs
    ):

        # get all init args and values and set
        local_params = {k: v for (k, v) in locals().items() if k not in ['kwargs', 'self']}
        kw_params = locals()['kwargs']

        self.set_params(**local_params)
        super().__init__(**kwargs)

        # associates with the *.stan model resource
        self.stan_model_name = "eswr"
        self.pyro_model_name = None
        self.min_nu = 5.0
        self.max_nu = 40.0
        self.num_of_regressor = 0


    def _set_computed_params(self):
        pass

    def _set_dynamic_inputs(self):
        if self.is_multiplicative:
            self.df = self._log_transform_df(self.df, do_fit=True)

        # a few of the following are related with training data.
        self.response = self.df[self.response_col].values
        self.num_of_observations = len(self.response)
        self.cauchy_sd = max(self.response) / 30.0
        self.response_sd = np.std(self.response)
        if self.regressor_col:
            self.num_of_regressor = len(self.regressor_col)
            self.regressor_matrix = self.df[self.regressor_col].values
        else:
            self.regressor_matrix = np.zeros((self.num_of_observations, 0))
        self._setup_stan_init()

    def _setup_stan_init(self):
        pass
        # # to use stan default, set self.stan_int to 'random'
        # self.stan_init = []
        # # use the seed so we can replicate results with same seed
        # np.random.seed(self.seed)
        # # ch is not used but we need the for loop to append init points across chains
        # for ch in range(self.chains):
        #     temp_init_dict = {}
        #     if self.seasonality > 1:
        #         # note that although seed fixed, points are different across chains
        #         init = np.random.normal(loc=0, scale=0.05, size=self.seasonality - 1)
        #         init[init > 1.0] = 1.0
        #         init[init < -1.0] = -1.0
        #         temp_init_dict['init_s'] = init
        #     self.stan_init.append(temp_init_dict)

    def _set_model_param_names(self):
        self.model_param_names = []
        self.model_param_names += [param.value for param in eswr.BaseSamplingParameters]

        # append seasonality param names
        if self.regressor_col:
            self.model_param_names += [param.value for param in eswr.RegressionParameters]


    def _log_transform_df(self, df, do_fit=False):
        # transform the response column
        if do_fit:
            # make sure values are > 0
            if np.any(df[self.response_col] <= 0):
                raise IllegalArgument('Response and Features must be a positive number')

            df[self.response_col] = df[self.response_col].apply(np.log)
        return df

    def _predict(self, df=None, include_error=False, decompose=False):
        """Vectorized version of prediction math"""

        ################################################################
        # Model Attributes
        ################################################################

        model = deepcopy(self._posterior_state)
        for k, v in model.items():
            model[k] = torch.from_numpy(v)

        # We can pull any arbitrary value from teh dictionary because we hold the
        # safe assumption: the length of the first dimension is always the number of samples
        # thus can be safely used to determine `num_sample`. If predict_method is anything
        # other than full, the value here should be 1
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        # # seasonality components
        # seasonality_levels = model.get(
        #     eswr.SeasonalitySamplingParameters.SEASONALITY_LEVELS.value)
        # seasonality_smoothing_factor = model.get(
        #     eswr.SeasonalitySamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        # )

        # trend components
        level_smoothing_factor = model.get(
            eswr.BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        residual_sigma = model.get(eswr.BaseSamplingParameters.RESIDUAL_SIGMA.value)
        local_level = model.get(eswr.BaseSamplingParameters.LOCAL_TREND_LEVELS.value)

        # global_trend_level = model.get(
        #     eswr.BaseSamplingParameters.GLOBAL_TREND_LEVEL.value).view(num_sample, )
        global_trend_slope = model.get(
            eswr.BaseSamplingParameters.GLOBAL_TREND_SLOPE.value).view(num_sample, )
        # in-sample global trend
        in_global_trend = model.get(eswr.BaseSamplingParameters.GLOBAL_TREND.value)

        ################################################################
        # Prediction Attributes
        ################################################################

        # get training df meta
        training_df_meta = self.training_df_meta
        # remove reference from original input
        df = df.copy()

        # for multiplicative model
        if self.is_multiplicative:
            df = self._log_transform_df(df, do_fit=False)

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
        local_level_len = local_level.shape[1]
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
                -(len(set(training_df_meta['date_array']) - set(prediction_df_meta['date_array'])))
            # time index for prediction start
            start = pd.Index(
                training_df_meta['date_array']).get_loc(prediction_df_meta['prediction_start'])

        # regressor_beta = self.get_regression_coefs()
        regressor_coef = model.get(eswr.RegressionParameters.REGRESSION_COEF.value)
        num_of_coef = regressor_coef.shape[-1]

        full_len = trained_len + n_forecast_steps

        ################################################################
        # Seasonality Component
        ################################################################

        # # calculate seasonality component
        # if self.seasonality > 1:
        #     if full_len <= seasonality_levels.shape[1]:
        #         seasonality_component = seasonality_levels[:, :full_len]
        #     else:
        #         seasonality_forecast_length = full_len - seasonality_levels.shape[1]
        #         init_zeros = torch.zeros((num_sample, seasonality_forecast_length), dtype=torch.double)
        #         seasonality_component = torch.cat((seasonality_levels, init_zeros), dim=1)

        ################################################################
        # Trend Component
        ################################################################

        # calculate level component.
        # However, if predicted end of period > training period, update with out-of-samples forecast
        if full_len <= local_level_len:
            global_trend = in_global_trend[:, :full_len]
            local_trend = local_level[:, :full_len]
            # # in-sample error are iids
            # if include_error:
            #     error_value = nct.rvs(
            #         df=residual_degree_of_freedom.unsqueeze(-1),
            #         nc=0,
            #         loc=0,
            #         scale=residual_sigma.unsqueeze(-1),
            #         size=(num_sample, full_len)
            #     )
            #
            #     error_value = torch.from_numpy(error_value.reshape(num_sample, full_len)).double()
            #     trend_component += error_value
        else:
            trend_forecast_length = full_len - local_level_len
            # additional initial zeros to append
            init_zeros = torch.zeros((num_sample, trend_forecast_length), dtype=torch.double)
            # in_local_trend = local_level
            # in-sample error are iids
            # if include_error:
            #     error_value = nct.rvs(
            #         df=residual_degree_of_freedom.unsqueeze(-1),
            #         nc=0,
            #         loc=0,
            #         scale=residual_sigma.unsqueeze(-1),
            #         size=(num_sample, local_global_trend_sums.shape[1])
            #     )
            #
            #     error_value = torch.from_numpy(
            #         error_value.reshape(num_sample, local_global_trend_sums.shape[1])).double()
            #     trend_component += error_value

            # trend_forecast_matrix = torch.zeros((num_sample, n_forecast_steps - 1), dtype=torch.double)
            local_trend = torch.cat((local_level, init_zeros), dim=1)
            global_trend = torch.cat((in_global_trend, init_zeros), dim=1)

            init_zeros = torch.zeros((num_sample, trend_forecast_length, ), dtype=torch.double)

            # regressor_beta = torch.cat((regressor_beta, init_zeros), dim=1)

            for idx in range(local_level_len, full_len):
                local_trend[:, idx] = local_trend[:, idx-1]
                regressor_coef[:, idx, :] = regressor_coef[:, idx-1, :]
                # global_trend[:, idx] = global_trend_slope * idx * self.time_delta
                global_trend[:, idx] = global_trend_slope * idx
                # if include_error:
                #     error_value = nct.rvs(
                #         df=residual_degree_of_freedom,
                #         nc=0,
                #         loc=0,
                #         scale=residual_sigma,
                #         size=num_sample
                #     )
                #     error_value = torch.from_numpy(error_value).double()
                #     trend_component[:, idx] += error_value

                # if self.seasonality > 1 and idx > seasonality_levels.shape[1]:
                #     seasonality_component[:, idx] = seasonality_component[:, idx - self.seasonality]
                    #     seasonality_component[:, idx + self.seasonality] = \
                #         seasonality_smoothing_factor.flatten() \
                #         * (trend_component[:, idx] + seasonality_component[:, idx] -
                #            new_local_trend_level) \
                #         + (1 - seasonality_smoothing_factor.flatten()) * seasonality_component[:, idx]

                # last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = local_trend[:, start:] + global_trend[:, start:]
        # seasonality_component = seasonality_component[:, start:]

        # sum components
        pred_array = trend_component + seasonality_component

        # for the multiplicative case
        if self.is_multiplicative:
            pred_array = (torch.exp(pred_array)).numpy()
            trend_component = (torch.exp(trend_component)).numpy()
            seasonality_component = (torch.exp(seasonality_component)).numpy()
        else:
            pred_array = pred_array.numpy()
            trend_component = trend_component.numpy()
            seasonality_component = seasonality_component.numpy()

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend_component,
                'seasonality': seasonality_component,
                # 'regression': regressor_component
            }

            return decomp_dict

        return {'prediction': pred_array}

    def get_regression_coefs(self):
        """Return DataFrame regression coefficients

        If PredictMethod is `full` return `mean` of coefficients instead
        """
        predict_method = PredictMethod.MEAN.value \
            if self.predict_method == PredictMethod.FULL_SAMPLING.value \
            else self.predict_method

        regressor_coefs = self.aggregated_posteriors\
            .get(predict_method)\
            .get(eswr.RegressionParameters.REGRESSION_COEF.value)

        return regressor_coefs