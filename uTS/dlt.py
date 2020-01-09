from uTS.stan_estimator import StanEstimator
from uTS.utils.constants import (
    LocalTrendStanSamplingParameters,
    SeasonalityStanSamplingParameters,
    DAMPEDTRENDStanSamplingParameters,
    RegressionStanSamplingParameters,
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA
)

from uTS.exceptions import (
    PredictionException,
    IllegalArgument
)

from uTS.utils.utils import is_ordered_datetime

import numpy as np
import pandas as pd
from scipy.stats import nct
import math as math


class DLT(StanEstimator):
    """Implementation of Damped-Local-Trend (LGT) model with seasonality.


    Prediction

    LGT follows state space decomposition such that predictions are sum of the three states--
    trend, seasonality and externality (regression). In math, we have

    .. math::
        \hat{y}_t=\mu_t+s_t+r_t

    Update Process

    Except externality, states are updated in a sequential manner.  In math, we have

    .. math::
        r_t=\{X\\beta\}_t

        \mu_t=l_{t-1}+\\theta_{loc}{b_{t-1}}+\\theta_{glb}{|l_{t-1}|^{\lambda}}

        l_t=\\alpha_{lev}(y_t-s_t-r_t)+(1-\\alpha_{lev})l_{t-1}

        b_t=\\alpha_{slp}(l_t-l_{t-1})+(1-\\alpha_{slp})b_{t-1}

        s_t=\\alpha_{sea}(y_t-l_t-r_t)+(1-\\alpha_{sea})s_{t-1}


    Likelihood & Prior

    .. math::
        y_t\sim{Stt(\hat{y},v,\sigma_{obs})}

        \\beta\sim{N(0,\sigma_{\\beta})}

        \sigma_{obs}\sim Cauchy(0, C_{scale})


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
    cauchy_sd : float
        scale parameter of prior of observation residuals scale parameter `C_scale`
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
    use_damped_trend : int
        binary input 0 for using LGT Model; 1 for using Damped Trend Model
    fix_regression_coef_sd : int
        binary input 0 for using point prior of regressors sigma; 1 for using Cauchy prior for regressor
        sigma
    regressor_sigma_sd : float
        scale parameter of prior of regressor coefficient scale parameter.
        Ignored when `fix_regression_coef_sd` is 1.
    damped_factor_fixed : float
        input between 0 and 1 which specify damped effect of local slope per period.
        Ignored when `use_damped_trend` is 0.
    damped_factor_min : float
         minimum value allowed for damped factor samples. Ignored when `damped_factor_fixed` > 0
    damped_factor_max : float
         maximum value allowed for damped factor  samples. Ignored when `damped_factor_fixed` > 0
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

    def __init__(
            self, regressor_col=None, regressor_sign=None,
            regressor_beta_prior=None, regressor_sigma_prior=None,
            cauchy_sd=None, min_nu=5, max_nu=40,
            seasonality=0, seasonality_min=-1.0, seasonality_max=1.0,
            seasonality_smoothing_min=0, seasonality_smoothing_max=1,
            level_smoothing_min=0, level_smoothing_max=1,
            slope_smoothing_min=0, slope_smoothing_max=1,
            use_damped_trend=0, damped_factor_min=0.8, damped_factor_max=0.999,
            regression_coef_max=1.0, fix_regression_coef_sd=1, regressor_sigma_sd=1.0,
            damped_factor_fixed=0.9, **kwargs
    ):

        # get all init args and values and set
        local_params = {k: v for (k, v) in locals().items() if k not in ['kwargs', 'self']}
        kw_params = locals()['kwargs']

        self.set_params(**local_params)
        super().__init__(**kwargs)

        # associates with the *.stan model resource
        self.stan_model_name = "dlt"

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

    def _set_dynamic_inputs(self):

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

        def _validate():
            if self.regressor_col is not None and \
                    not set(self.regressor_col).issubset(self.df.columns):
                raise IllegalArgument(
                    "DataFrame does not contain specified regressor colummn(s)."
                )

        _validate()

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
        self.model_param_names += [param for param in LocalTrendStanSamplingParameters]

        # append seasonality param names
        if self.seasonality > 1:
            self.model_param_names += [param for param in SeasonalityStanSamplingParameters]

        # append damped trend param names
        if self.damped_factor_fixed < 0:
            self.model_param_names += [param for param in DAMPEDTRENDStanSamplingParameters]

        # append positive regressors if any
        if self.num_of_positive_regressors > 0:
            self.model_param_names += [RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA]

        # append regular regressors if any
        if self.num_of_regular_regressors > 0:
            self.model_param_names += [RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA]

    def _predict_once(self, df=None, include_error=False, decompose=False):

        ################################################################
        # Model Attributes
        ################################################################

        # get model attributes
        model = self._posterior_state

        # seasonality components
        seasonality_levels = model.get(SeasonalityStanSamplingParameters.SEASONALITY_LEVELS)
        seasonality_smoothing_factor = model.get(
            SeasonalityStanSamplingParameters.SEASONALITY_SMOOTHING_FACTOR
        )

        # trend components
        slope_smoothing_factor = model.get(LocalTrendStanSamplingParameters.SLOPE_SMOOTHING_FACTOR)
        level_smoothing_factor = model.get(LocalTrendStanSamplingParameters.LEVEL_SMOOTHING_FACTOR)
        local_global_trend_sums = model.get(LocalTrendStanSamplingParameters.LOCAL_GLOBAL_TREND_SUMS)
        local_trend_levels = model.get(LocalTrendStanSamplingParameters.LOCAL_TREND_LEVELS)
        local_trend_slopes = model.get(LocalTrendStanSamplingParameters.LOCAL_TREND_SLOPES)
        residual_degree_of_freedom = model.get(LocalTrendStanSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM)
        residual_sigma = model.get(LocalTrendStanSamplingParameters.RESIDUAL_SIGMA)

        if self.damped_factor_fixed > 0:
            damped_factor = self.damped_factor_fixed
        else:
            damped_factor = model.get(DAMPEDTRENDStanSamplingParameters.DAMPED_FACTOR)

        # regression components
        pr_beta = model.get(RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA, np.array([]))
        rr_beta = model.get(RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA, np.array([]))
        regressor_beta = np.concatenate((pr_beta, rr_beta))

        ################################################################
        # Prediction Attributes
        ################################################################

        # get training df meta
        training_df_meta = self.training_df_meta

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
            regressor_matrix = df[self.regressor_col]
            regressor_component = regressor_matrix.dot(regressor_beta).values
        else:
            # regressor is always dependent with df. hence, no need to make full size
            regressor_component = np.zeros(output_len)

        ################################################################
        # Seasonality Component
        ################################################################

        # calculate seasonality component
        if self.seasonality > 1:
            if full_len <= len(seasonality_levels):
                seasonality_component = seasonality_levels[:full_len]
            else:
                seasonality_component = np.concatenate((
                    seasonality_levels,
                    np.zeros(full_len - len(seasonality_levels)),
                ))
        else:
            seasonality_component = np.zeros(full_len)

        ################################################################
        # Trend Component
        ################################################################

        # calculate level component.
        # However, if predicted end of period > training period, update with out-of-samples forecast
        if full_len <= trained_len:
            trend_component = local_global_trend_sums[:full_len]
        else:
            trend_component = np.concatenate((
                local_global_trend_sums, np.zeros(full_len - trained_len),
            ))

            last_local_trend_level = local_trend_levels[-1]
            last_local_trend_slope = local_trend_slopes[-1]

            for idx in range(trained_len, full_len):
                # based on model, split cases for trend update
                current_local_trend = damped_factor * last_local_trend_slope
                trend_component[idx] = last_local_trend_level + current_local_trend

                if include_error:
                    error_value = nct.rvs(
                        df=residual_degree_of_freedom,
                        nc=0,
                        loc=0,
                        scale=residual_sigma,
                        size=1
                    )[0]  # scalar value
                    trend_component[idx] += error_value

                trend_component[idx] = max(trend_component[idx], 0)

                new_local_trend_level = \
                    level_smoothing_factor * trend_component[idx] \
                    + (1 - level_smoothing_factor) * last_local_trend_level
                last_local_trend_slope = \
                    slope_smoothing_factor * (new_local_trend_level -
                                                       last_local_trend_level) \
                    + (1 - slope_smoothing_factor) * damped_factor * last_local_trend_slope

                if self.seasonality > 1 and idx + self.seasonality < full_len:
                    seasonality_component[idx + self.seasonality] = \
                        seasonality_smoothing_factor \
                        * (trend_component[idx] + seasonality_component[idx] -
                           new_local_trend_level) \
                        + (1 - seasonality_smoothing_factor) * seasonality_component[idx]

                last_local_trend_level = new_local_trend_level

        ################################################################
        # Combine Components
        ################################################################

        # trim component with right start index
        trend_component = trend_component[start:]
        seasonality_component = seasonality_component[start:]

        # sum components
        pred_array = trend_component + seasonality_component + regressor_component

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
