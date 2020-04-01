import numpy as np
import pandas as pd
from scipy.stats import nct
import torch
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

from orbit.estimator import Estimator
from orbit.constants import lgt
from orbit.constants.constants import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA
)
from orbit.exceptions import (
    PredictionException,
    IllegalArgument
)
from orbit.utils.utils import is_ordered_datetime


class LGT(Estimator):
    """Implementation of Local-Global-Trend (LGT) model with seasonality.

    **Evaluation Process & Likelihood**

    LGT follows state space decomposition such that predictions are sum of the three states--
    trend, seasonality and externality (regression).

    .. math::
        \hat{y}_t=\mu_t+s_t+r_t

        y - \hat{y} \sim \\text{Student-T}(\\nu, 0,\sigma)

        \sigma \sim \\text{Half-Cauchy}(0, \gamma)

    **Update Process**

    States are updated in a sequential manner except externality.

    .. math::
        r_t=\{X\\beta\}_t

        \mu_t=l_{t-1}+\\theta_{loc}{b_{t-1}}+\\theta_{glb}{|l_{t-1}|^{\lambda}}

        l_t=\\rho_{l}(y_t-s_t-r_t)+(1-\\rho_{l})l_{t-1}

        b_t=\\rho_{b}(l_t-l_{t-1})+(1-\\rho_{b}){b_{t-1}}

        s_t=\\rho_{s}(y_t-l_t-r_t)+(1-\\rho_{s})s_{t-1}


    **Priors**

    .. math::
        \\beta\sim{N(0,\sigma_{\\beta})}

        \\rho_{l},\\rho_{b},\\rho_{s} \sim \\text{Uniform}(0,1)

        \\theta_{loc}, \\theta_{glb}, \lambda, \\nu \\text{ also follow uniform priors with}

    different bounds.

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
    regression_coef_max : float
        Maximum absolute value allowed for regression coefficient samples


    Notes
    -----
    LGT model is an extensions of traditional exponential smoothing models. For details, see
    https://otexts.com/fpp2/holt-winters.html.
    In short, the model follows state space decomposition such that predictions are the
    sum of the three states--trend, seasonality and externality (regression).  The states updates
    sequentially with smoothing, scale and other parameters.

    The original model was created by Slawek Smyl, Christoph Bergmeir, Erwin Wibowo,
    To Wang(Edwin) Ng. For details, see https://cran.r-project.org/web/packages/Rlgt/index.html

    We re-parameterized the model to reduce complexity and to be in additive form so that user has
    an option to transform the model back to multiplicative form through logarithm
    transformation.  These changes enhance  speed and stability of sampling. The model also provides
    more support on external regressors.

    One may note that there is a small but critical difference of the formula in both training (*.stan file) and
    prediction (predict()) such that the `l(t)` is updated with levels only l(t-1) rather than
    the integrated trend (like l(t-1) + b(t-1) in traditional exp. smoothing models).
    """
    # this must be defined in child class
    _stan_input_mapper = lgt.StanInputMapper

    def __init__(
            self, regressor_col=None, regressor_sign=None,
            regressor_beta_prior=None, regressor_sigma_prior=None,
            is_multiplicative=True, auto_scale=False, cauchy_sd=None, min_nu=5, max_nu=40,
            seasonality=0, seasonality_min=-1.0, seasonality_max=1.0,
            seasonality_smoothing_min=0, seasonality_smoothing_max=1.0,
            global_trend_coef_min=-0.5, global_trend_coef_max=0.5,
            global_trend_pow_min=0, global_trend_pow_max=1,
            local_trend_coef_min=0, local_trend_coef_max=1,
            level_smoothing_min=0, level_smoothing_max=1,
            slope_smoothing_min=0, slope_smoothing_max=1,
            regression_coef_max=1.0, fix_regression_coef_sd=1, regressor_sigma_sd=1.0,
            **kwargs
    ):

        # get all init args and values and set
        local_params = {k: v for (k, v) in locals().items() if k not in ['kwargs', 'self']}
        kw_params = locals()['kwargs']

        self.set_params(**local_params)
        super().__init__(**kwargs)

        # associates with the *.stan model resource
        self.stan_model_name = "lgt"
        self.pyro_model_name = "orbit.pyro.lgt.LGTModel"
        # rescale depends on max of response
        self.response_min_max_scaler = None
        # rescale depends on num of regressors
        self.regressor_min_max_scaler = None
        if auto_scale and not is_multiplicative:
            raise IllegalArgument(
                'Auto-scale is not supported for additive model. Please turn off auto-scale.'
            )

    def _set_computed_params(self):
        self._setup_computed_regression_params()
        self._setup_seasonality_init()

    def _setup_computed_regression_params(self):

        def _validate(regression_params, valid_length):
            for p in regression_params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

        # start with defaults
        self.num_of_positive_regressors = 0
        self.positive_regressor_col = []
        self.positive_regressor_beta_prior = []
        self.positive_regressor_sigma_prior = []

        self.num_of_regular_regressors = 0
        self.regular_regressor_col = []
        self.regular_regressor_beta_prior = []
        self.regular_regressor_sigma_prior = []

        # if no regressors, end here
        if self.regressor_col is None:
            return

        num_of_regressors = len(self.regressor_col)

        _validate(
            [self.regressor_sign, self.regressor_beta_prior, self.regressor_sigma_prior],
            num_of_regressors
        )

        if self.regressor_sign is None:
            self.regressor_sign = [DEFAULT_REGRESSOR_SIGN] * num_of_regressors

        if self.regressor_beta_prior is None:
            self.regressor_beta_prior = [DEFAULT_REGRESSOR_BETA] * num_of_regressors

        if self.regressor_sigma_prior is None:
            self.regressor_sigma_prior = [DEFAULT_REGRESSOR_SIGMA] * num_of_regressors

        # inside *.stan files, we need to distinguish regular regressors from positive regressors
        for index, reg_sign in enumerate(self.regressor_sign):
            if reg_sign == '+':
                self.num_of_positive_regressors += 1
                self.positive_regressor_col.append(self.regressor_col[index])
                self.positive_regressor_beta_prior.append(self.regressor_beta_prior[index])
                self.positive_regressor_sigma_prior.append(self.regressor_sigma_prior[index])
            else:
                self.num_of_regular_regressors += 1
                self.regular_regressor_col.append(self.regressor_col[index])
                self.regular_regressor_beta_prior.append(self.regressor_beta_prior[index])
                self.regular_regressor_sigma_prior.append(self.regressor_sigma_prior[index])

    def _scale_df(self, df, do_fit=False):
        regression_sigma_sum = 0.0
        # scale regressors if avaliable
        if self.regressor_col is not None:
            regression_sigma_sum = np.sum(self.regressor_sigma_prior)
            # fit regerssor scaler in fitting
            if do_fit:
                self.regressor_min_max_scaler = MinMaxScaler(1, 2.719)
                df[self.regressor_col] = \
                    self.regressor_min_max_scaler.fit_transform(df[self.regressor_col])
            # transfrom regressors
            else:
                df[self.regressor_col] = \
                    self.regressor_min_max_scaler.transform(df[self.regressor_col])

        # fit response scaler in fitting
        if do_fit:
            n = df.shape[0]
            x = df[self.response_col].astype(np.float64).values.reshape(n, 1)
            # bounded by chance of seasoanlity and sum of regression causing negative
            # it won't completely get rid of chacne but should catch most of the cases
            lower = max(1.001 +
                        max(-1 * self.seasonality_min, 0) +
                        2 * regression_sigma_sum, np.min(x))
            upper = min(lower + 10, np.max(x))
            self.response_min_max_scaler = MinMaxScaler((lower, upper))
            df[self.response_col] = self.response_min_max_scaler.fit_transform(x).flatten()
        return df

    def _log_transform_df(self, df, do_fit=False):
        # transform the response column
        if do_fit:
            data_cols = [self.response_col] + self.regressor_col \
                if self.regressor_col is not None \
                else [self.response_col]
            # make sure values are > 0
            if np.any(df[data_cols] <= 0):
                raise IllegalArgument('Response and Features must be a positive number')

            df[self.response_col] = df[self.response_col].apply(np.log)

        # transform the regressor columns if exist
        if self.regressor_col is not None:
            # make sure values are > 0
            if np.any(df[self.regressor_col] <= 0):
                raise IllegalArgument('Features must be a positive number')

            df[self.regressor_col] = df[self.regressor_col].apply(np.log)

        return df

    def _set_dynamic_inputs(self):

        # validate regression columns
        def _validate_regression_columns():
            if self.regressor_col is not None and \
                    not set(self.regressor_col).issubset(self.df.columns):
                raise IllegalArgument(
                    "DataFrame does not contain specified regressor colummn(s)."
                )

        _validate_regression_columns()

        if self.auto_scale:
            self.df = self._scale_df(self.df, do_fit=True)
        if self.is_multiplicative:
            self.df = self._log_transform_df(self.df, do_fit=True)

        # a few of the following are related with training data.
        self.response = self.df[self.response_col].values
        self.num_of_observations = len(self.response)
        self.cauchy_sd = max(
                self.response,
            ) / 300 if self.cauchy_sd is None else self.cauchy_sd

        self._setup_regressor_inputs()

    def _setup_seasonality_init(self):
        if self.seasonality > 1:
            # use the seed so we can replicate results with same seed
            np.random.seed(self.seed)
            # replace with empty list from purely 'random'
            self.stan_init = []
            # ch is not used but we need the for loop to append init points across chains
            for ch in range(self.chains):
                temp_init = {}
                # note that although seed fixed, points are different across chains
                seas_init = np.random.normal(loc=0, scale=0.05, size=self.seasonality - 1)
                seas_init[seas_init > self.seasonality_max] = self.seasonality_max
                seas_init[seas_init < self.seasonality_min] = self.seasonality_min
                temp_init['init_sea'] = seas_init
                self.stan_init.append(temp_init)

    def _setup_regressor_inputs(self):

        self.positive_regressor_matrix = np.zeros((self.num_of_observations, 0))
        self.regular_regressor_matrix = np.zeros((self.num_of_observations, 0))

        # update regression matrices
        if self.num_of_positive_regressors > 0:
            self.positive_regressor_matrix = self.df.filter(
                items=self.positive_regressor_col,).values

        if self.num_of_regular_regressors > 0:
            self.regular_regressor_matrix = self.df.filter(
                items=self.regular_regressor_col,).values

    def _set_model_param_names(self):
        self.model_param_names = []
        self.model_param_names += [param.value for param in lgt.BaseStanSamplingParameters]

        # append seasonality param names
        if self.seasonality > 1:
            self.model_param_names += [param.value for param in lgt.SeasonalityStanSamplingParameters]

        # append positive regressors if any
        if self.num_of_positive_regressors > 0:
            self.model_param_names += [
                lgt.RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA.value]

        # append regular regressors if any
        if self.num_of_regular_regressors > 0:
            self.model_param_names += [
                lgt.RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA.value]

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

        # seasonality components
        seasonality_levels = model.get(
            lgt.SeasonalityStanSamplingParameters.SEASONALITY_LEVELS.value)
        seasonality_smoothing_factor = model.get(
            lgt.SeasonalityStanSamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend components
        slope_smoothing_factor = model.get(
            lgt.BaseStanSamplingParameters.SLOPE_SMOOTHING_FACTOR.value)
        level_smoothing_factor = model.get(
            lgt.BaseStanSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        local_trend_levels = model.get(lgt.BaseStanSamplingParameters.LOCAL_TREND_LEVELS.value)
        local_trend_slopes = model.get(lgt.BaseStanSamplingParameters.LOCAL_TREND_SLOPES.value)
        residual_degree_of_freedom = model.get(
            lgt.BaseStanSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM.value)
        residual_sigma = model.get(lgt.BaseStanSamplingParameters.RESIDUAL_SIGMA.value)

        local_trend_coef = model.get(lgt.BaseStanSamplingParameters.LOCAL_TREND_COEF.value)
        global_trend_power = model.get(lgt.BaseStanSamplingParameters.GLOBAL_TREND_POWER.value)
        global_trend_coef = model.get(lgt.BaseStanSamplingParameters.GLOBAL_TREND_COEF.value)
        local_global_trend_sums = model.get(
            lgt.BaseStanSamplingParameters.LOCAL_GLOBAL_TREND_SUMS.value)

        # regression components
        pr_beta = model.get(lgt.RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA.value)
        rr_beta = model.get(lgt.RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA.value)
        if pr_beta is not None and rr_beta is not None:
            pr_beta = pr_beta if len(pr_beta.shape) == 2 else pr_beta.reshape(1, -1)
            rr_beta = rr_beta if len(rr_beta.shape) == 2 else rr_beta.reshape(1, -1)
            regressor_beta = torch.cat((pr_beta, rr_beta), dim=1)
        else:
            regressor_beta = pr_beta or rr_beta

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
                -(len(set(training_df_meta['date_array']) - set(prediction_df_meta['date_array'])))
            # time index for prediction start
            start = pd.Index(
                training_df_meta['date_array']).get_loc(prediction_df_meta['prediction_start'])

        full_len = trained_len + n_forecast_steps

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
                seasonality_forecast_matrix \
                    = torch.zeros((num_sample, seasonality_forecast_length), dtype=torch.double)
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
            trend_component = local_global_trend_sums[:, :full_len]
        else:
            trend_forecast_length = full_len - trained_len
            trend_forecast_matrix \
                = torch.zeros((num_sample, trend_forecast_length), dtype=torch.double)
            trend_component = torch.cat((local_global_trend_sums, trend_forecast_matrix), dim=1)

            last_local_trend_level = local_trend_levels[:, -1]
            last_local_trend_slope = local_trend_slopes[:, -1]

            trend_component_zeros = torch.zeros_like(trend_component[:, 0])

            for idx in range(trained_len, full_len):
                current_local_trend = local_trend_coef.flatten() * last_local_trend_slope
                global_trend_power_term = torch.pow(
                    torch.abs(last_local_trend_level),
                    global_trend_power.flatten()
                )
                current_global_trend = global_trend_coef.flatten() * global_trend_power_term
                trend_component[:, idx] \
                    = last_local_trend_level + current_local_trend + current_global_trend

                if include_error:
                    error_value = nct.rvs(
                        df=residual_degree_of_freedom,
                        nc=0,
                        loc=0,
                        scale=residual_sigma,
                        size=num_sample
                    )
                    error_value = torch.from_numpy(error_value).double()
                    trend_component[:, idx] += error_value

                # a 2d tensor of size (num_sample, 2) in which one of the elements
                # is always zero. We can use this to torch.max() across the sample dimensions
                trend_component_augmented = torch.cat(
                    (trend_component[:, idx][:, None], trend_component_zeros[:, None]), dim=1)

                max_value, _ = torch.max(trend_component_augmented, dim=1)

                trend_component[:, idx] = max_value

                new_local_trend_level = \
                    level_smoothing_factor * trend_component[:, idx] \
                    + (1 - level_smoothing_factor) * last_local_trend_level

                last_local_trend_slope = \
                    slope_smoothing_factor * (new_local_trend_level - last_local_trend_level) \
                    + (1 - slope_smoothing_factor) * last_local_trend_slope

                if self.seasonality > 1 and idx + self.seasonality < full_len:
                    seasonality_component[:, idx + self.seasonality] = \
                        seasonality_smoothing_factor.flatten() \
                        * (trend_component[:, idx] + seasonality_component[:, idx] -
                           new_local_trend_level) \
                        + (1 - seasonality_smoothing_factor.flatten()) * seasonality_component[:, idx]

                last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = trend_component[:, start:]
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

    def _validate_params(self):
        pass
