import pandas as pd
import numpy as np
from scipy.stats import nct
import torch
from copy import deepcopy

from orbit.lgt import LGT
from orbit.constants import dlt
from orbit.exceptions import (
    PredictionException,
    IllegalArgument
)
from orbit.utils.utils import is_ordered_datetime


class DLT(LGT):
    """Implementation of Damped-Local-Trend (LGT) model with seasonality.

    **Evaluation Process & Likelihood**

    DLT follows state space decomposition such that predictions are sum of the three states--
    trend, seasonality and externality (regression).

    .. math::
        \hat{y}_t=\mu_t+s_t+r_t

        y - \hat{y} \sim \\text{Student-T}(\\nu, 0,\sigma)

        \sigma \sim \\text{Half-Cauchy}(0, \gamma)

    **Update Process**

    States are updated in a sequential manner except externality and global trend.

    .. math::
        r_t=\{X\\beta\}_t

        \mu_t=g_{t} + l_{t-1} + \\delta{b_{t-1}}

        l_t=\\rho_{l}(y_t-s_t-r_t)+(1-\\rho_{l})l_{t-1}

        b_t=\\rho_{b}(l_t-l_{t-1})+(1-\\rho_{b})\\delta{b_{t-1}}

        s_t=\\rho_{s}(y_t-l_t-r_t)+(1-\\rho_{s})s_{t-1}

    **Priors**

    .. math::
        \\beta\sim{N(0,\sigma_{\\beta})}

        \\rho_{l},\\rho_{b},\\rho_{s} \sim \\text{Uniform}(0,1)


    **Notes**

    *Global Trend*

    .. math::
       g_t \\text{ is modeled as deterministic trend with three options:}

    1. linear
    2. log-linear
    3. logistic

    *Damped Factor*

    .. math::
        \\delta \\text{ is capture damped effect of the local trend.}

    As default, it is an input from user. There is an option for user to model it as
    a parameter in a uniform range.

    Parameters
    ----------
    regressor_col : list
        a list of regressor column names in the dataframe
    regressor_sign : list
        a list of either {'+', '='} to determine if regression estimates should be restricted
        to positive weights only. The length of `regressor_sign` must be the same as
        `regressor_col`
    regressor_beta_prior : list
        prior values for regressor betas. The length of `regressor_beta_prior` must be the
        same as `regressor_col`
    regressor_sigma_prior : list
        prior values for regressors standard deviation. The length of `regressor_sigma_prior` must
        be the same as `regressor_col`
    is_multiplicative : bool
        if True, response and regressor values are log transformed such that the model is
        multiplicative. If False, no transformations are applied. Default True.
    auto_scale : bool
        **EXPERIMENTAL AND UNSTABLE** if True, response and regressor values are transformed
        with a `MinMaxScaler` such that the min value is `e` and max value is
        the max value in the data
    cauchy_sd : float
        a hyperprior for residuals scale parameter
    seasonality : int
        The length of the seasonality period. For example, if dataframe contains weekly data
        points, `52` would indicate a yearly seasonality.
    seasonality_min : float
        minimum value allowed for initial seasonality samples
    seasonality_max : float
        maximum value allowed for initial seasonality samples
    seasonality_smoothing_min : float
        minimum value allowed for seasonality smoothing coefficient samples
    seasonality_smoothing_max : float
        maximum value allowed for seasonality smoothing coefficient samples
    global_trend_coef_min : float
        minimum value allowed for global trend coefficient samples
    global_trend_coef_max : float
        maximum value allowed for global trend coefficient samples
    global_trend_pow_min : float
        minimum value allowed for global trend power samples
    global_trend_pow_max : float
        maximum value allowed for global trend power samples
    local_trend_coef_min : float
        minimum value allowed for local trend coefficient samples
    local_trend_coef_max : float
        maximum value allowed for local trend coefficient samples
    level_smoothing_min : float
         minimum value allowed for level smoothing coefficient samples
    level_smoothing_max : float
        maximum value allowed for level smoothing coefficient samples
    slope_smoothing_min : float
        minimum value allowed for local slope smoothing coefficient samples
    slope_smoothing_max : float
        maximum value allowed for local slope smoothing coefficient samples
    fix_regression_coef_sd : int
        binary input 1 for using point prior of regressors sigma; 0 for using Cauchy hyperprior for
        regressor sigma
    regressor_sigma_sd : float
        hyperprior for regressor coefficient scale parameter. Ignored when `fix_regression_coef_sd`
        is 1.
    damped_factor_fixed : float
        input between 0 and 1 which specify damped effect of local slope per period.
    damped_factor_min : float
         minimum value allowed for damped factor samples. Ignored when `damped_factor_fixed` > 0
    damped_factor_max : float
         maximum value allowed for damped factor  samples. Ignored when `damped_factor_fixed` > 0
    regression_coef_max : float
        Maximum absolute value allowed for regression coefficient samples


    Notes
    -----
    DLT provides another option for exponential smoothing models. Unlike LGT, the additive model
    allows any real value of response while the multiplicative version requires non-negative
    value.  Moreover, it provides damped trend for user to perform long-term forecast.
    """
    # this must be defined in child class
    _stan_input_mapper = dlt.StanInputMapper

    def __init__(
            self, regressor_col=None, regressor_sign=None,
            regressor_beta_prior=None, regressor_sigma_prior=None,
            is_multiplicative=True, auto_scale=False, cauchy_sd=None, min_nu=5, max_nu=40,
            seasonality=0, seasonality_min=-1.0, seasonality_max=1.0,
            seasonality_smoothing_min=0, seasonality_smoothing_max=1,
            level_smoothing_min=0, level_smoothing_max=1,
            slope_smoothing_min=0, slope_smoothing_max=1,
            lasso_scale=0.1, auto_ridge_scale=0.1, regression_penalty='fixed_ridge',
            damped_factor_min=0.8, damped_factor_max=1,
            global_trend_option='linear',
            damped_factor_fixed=0.8, **kwargs
    ):

        # get all init args and values and set
        local_params = {k: v for (k, v) in locals().items() if k not in ['kwargs', 'self']}
        kw_params = locals()['kwargs']

        self.set_params(**local_params)
        super(LGT, self).__init__(**kwargs)  # note this is the base class

        # associates with the *.stan model resource
        self.stan_model_name = "dlt"
        # self.pyro_model_name = "orbit.pyro.dlt.DLTModel--WIP"
        if not hasattr(dlt.GlobalTrendOption, global_trend_option):
            gt_options = [e.name for e in dlt.GlobalTrendOption]
            raise IllegalArgument("global_trend_option must be one of these {}".format(gt_options))

        self._global_trend_option = getattr(dlt.GlobalTrendOption, global_trend_option).value
        self._regression_penalty = getattr(dlt.RegressionPenalty, regression_penalty).value

    def _set_model_param_names(self):
        self.model_param_names = []
        self.model_param_names += [param.value for param in dlt.BaseSamplingParameters]

        # append seasonality param names
        if self.seasonality > 1:
            self.model_param_names += [param.value for param in dlt.SeasonalityStanSamplingParameters]

        # append damped trend param names
        if self.damped_factor_fixed < 0:
            self.model_param_names += [
                param.value for param in dlt.DampedTrendStanSamplingParameters]

        # append positive regressors if any
        if self.num_of_positive_regressors > 0:
            self.model_param_names += [
                dlt.RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA.value]

        # append regular regressors if any
        if self.num_of_regular_regressors > 0:
            self.model_param_names += [
                dlt.RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA.value]

        if self._global_trend_option != dlt.GlobalTrendOption.flat.value:
            self.model_param_names += [param.value for param in dlt.GlobalTrendSamplingParameters]

    def _predict(self, df=None, include_error=False, decompose=False):

        ################################################################
        # Prediction Attributes
        ################################################################

        # get training df meta
        training_df_meta = self.training_df_meta
        # remove reference from original input
        df = df.copy()
        if self.auto_scale:
            df = self._scale_df(df, do_fit=False)
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
                               -(len(set(training_df_meta['date_array']) - set(
                                   prediction_df_meta['date_array'])))
            # time index for prediction start
            start = pd.Index(
                training_df_meta['date_array']).get_loc(prediction_df_meta['prediction_start'])

        full_len = trained_len + n_forecast_steps


        ################################################################
        # Model Attributes
        ################################################################

        # get model attributes
        model = deepcopy(self._posterior_state)

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
            dlt.SeasonalityStanSamplingParameters.SEASONALITY_LEVELS.value)
        seasonality_smoothing_factor = model.get(
            dlt.SeasonalityStanSamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend components
        slope_smoothing_factor = model.get(
            dlt.BaseSamplingParameters.SLOPE_SMOOTHING_FACTOR.value)
        level_smoothing_factor = model.get(
            dlt.BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        local_trend_levels = model.get(dlt.BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        local_trend_slopes = model.get(dlt.BaseSamplingParameters.LOCAL_TREND_SLOPES.value)
        local_trend = model.get(dlt.BaseSamplingParameters.LOCAL_TREND.value)
        residual_degree_of_freedom = model.get(
            dlt.BaseSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM.value)
        residual_sigma = model.get(dlt.BaseSamplingParameters.RESIDUAL_SIGMA.value)

        # set an additional attribute for damped factor when it is fixed
        # get it through user input field
        if self.damped_factor_fixed > 0:
            damped_factor = torch.empty(num_sample, dtype=torch.double)
            damped_factor.fill_(self.damped_factor_fixed)
        else:
            damped_factor = model.get(dlt.DampedTrendStanSamplingParameters.DAMPED_FACTOR.value)

        if self._global_trend_option != dlt.GlobalTrendOption.flat.value:
            global_trend_level = model.get(dlt.GlobalTrendSamplingParameters.GLOBAL_TREND_LEVEL.value).view(num_sample, )
            global_trend_slope = model.get(dlt.GlobalTrendSamplingParameters.GLOBAL_TREND_SLOPE.value).view(num_sample, )
            global_trend = model.get(dlt.GlobalTrendSamplingParameters.GLOBAL_TREND.value)
        else:
            global_trend = torch.zeros((num_sample, trained_len), dtype=torch.double)

        # regression components
        pr_beta = model.get(dlt.RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA.value)
        rr_beta = model.get(dlt.RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA.value)
        regressor_beta = self._concat_regression_coefs(pr_beta, rr_beta)


        ################################################################
        # Regression Component
        ################################################################

        # calculate regression component
        if self.regressor_col is not None and len(self.regular_regressor_col) > 0:
            regressor_beta = regressor_beta.t()
            regressor_matrix = df[self.regressor_col].values
            regressor_torch = torch.from_numpy(regressor_matrix)
            regressor_component = torch.matmul(regressor_torch, regressor_beta)
            regressor_component = regressor_component.t()
        else:
            # regressor is always dependent with df. hence, no need to make full size
            regressor_component = torch.zeros((num_sample, output_len), dtype=torch.double)

        ################################################################
        # Seasonality Component
        ################################################################

        # calculate seasonality component
        if self.seasonality > 1:
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
        else:
            trend_forecast_length = full_len - trained_len
            trend_forecast_init = torch.zeros((num_sample, trend_forecast_length), dtype=torch.double)
            full_local_trend = torch.cat((local_trend[:, :full_len], trend_forecast_init), dim=1)
            full_global_trend = torch.cat((global_trend[:, :full_len], trend_forecast_init), dim=1)
            last_local_trend_level = local_trend_levels[:, -1]
            last_local_trend_slope = local_trend_slopes[:, -1]

            for idx in range(trained_len, full_len):
                # based on model, split cases for trend update
                curr_local_trend = \
                    last_local_trend_level + damped_factor.flatten() * last_local_trend_slope
                full_local_trend[:, idx] = curr_local_trend
                # idx = time - 1
                if self.global_trend_option == dlt.GlobalTrendOption.linear.name:
                    full_global_trend[:, idx] = global_trend_level + global_trend_slope * idx
                elif self.global_trend_option == dlt.GlobalTrendOption.loglinear.name:
                    full_global_trend[:, idx] = global_trend_level + torch.log(1 + global_trend_slope * idx)
                elif self.global_trend_option == dlt.GlobalTrendOption.logistic.name:
                    full_global_trend[:, idx] = global_trend_level / (1 + torch.exp(-1 * global_trend_slope * idx))
                elif self._global_trend_option == dlt.GlobalTrendOption.flat.name:
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
                    # for convenience, we lump error on local trend since the formula would
                    # yield the same as yhat + noise - global_trend - seasonality - regression
                    # equivalent with local_trend + noise
                    full_local_trend[:, idx] += error_value

                # now full_local_trend contains the error term and hence we need to use
                # curr_local_trend as a proxy of previous level index
                new_local_trend_level = \
                    level_smoothing_factor * full_local_trend[:, idx] \
                    + (1 - level_smoothing_factor) * curr_local_trend
                last_local_trend_slope = \
                    slope_smoothing_factor * (new_local_trend_level - last_local_trend_level) \
                    + (1 - slope_smoothing_factor) * damped_factor.flatten() * last_local_trend_slope

                if self.seasonality > 1 and idx + self.seasonality < full_len:
                    seasonality_component[:, idx + self.seasonality] = \
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

        # for the multiplicative case
        if self.is_multiplicative:
            pred_array = (torch.exp(pred_array)).numpy()
            trend_component = (torch.exp(trend_component)).numpy()
            seasonality_component = (torch.exp(seasonality_component)).numpy()
            regressor_component = (torch.exp(regressor_component)).numpy()
        else:
            pred_array = pred_array.numpy()
            trend_component = trend_component.numpy()
            seasonality_component = seasonality_component.numpy()
            regressor_component = regressor_component.numpy()

        if self.auto_scale:
            # work around response_min_max_scaler initial shape
            init_shape = pred_array.shape
            # enfroce a 2D array
            pred_array = np.reshape(pred_array, (-1, 1))
            pred_array = self.response_min_max_scaler.inverse_transform(pred_array)
            pred_array = pred_array.reshape(init_shape)
            # we assume the unit is based on trend component while others are multipliers
            trend_component = self.response_min_max_scaler.inverse_transform(trend_component)

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

