import pandas as pd
import numpy as np
from scipy.stats import nct
import torch
from copy import copy, deepcopy

from ..constants import lgt
from ..constants.constants import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA,
    COEFFICIENT_DF_COLS,
    PredictMethod
)
from ..estimators.stan_estimator import StanEstimatorMCMC
from ..exceptions import IllegalArgument, LGTException, PredictionException
from ..utils.general import is_ordered_datetime


# todo: docstrings for LGT model


class BaseLGT(object):
    """Base LGT model object with shared functionality for Full, Aggregated, and MAP methods"""
    _data_input_mapper = lgt.DataInputMapper
    # stan model name (e.g. name of `*.stan` file in package)
    _stan_model_name = 'lgt'

    def __init__(self, response_col='y', date_col='ds', regressor_col=None,
                 seasonality=-1, period=1., is_multiplicative=True,
                 regressor_sign=None, regressor_beta_prior=None, regressor_sigma_prior=None,
                 regression_penalty='fixed_ridge', lasso_scale=0.5, auto_ridge_scale=0.5,
                 seasonality_sm_input=-1, slope_sm_input=-1, level_sm_input=-1,
                 estimator_type=StanEstimatorMCMC, **kwargs):
        self.response_col = response_col
        self.date_col = date_col
        self.regressor_col = regressor_col
        self.seasonality = seasonality
        self.period = period
        self.is_multiplicative = is_multiplicative
        self.regressor_sign = regressor_sign
        self.regressor_beta_prior = regressor_beta_prior
        self.regressor_sigma_prior = regressor_sigma_prior
        self.regression_penalty = regression_penalty
        self.lasso_scale = lasso_scale
        self.auto_ridge_scale = auto_ridge_scale
        self.seasonality_sm_input = seasonality_sm_input
        self.slope_sm_input = slope_sm_input
        self.level_sm_input = level_sm_input
        self.estimator_type = estimator_type

        # create concrete estimator object
        self.estimator = self.estimator_type(**kwargs)

        # init static data attributes
        # the following are set by `_set_static_data_attributes()`
        self._training_df_meta = None
        self._model_param_names = list()
        self._stan_data_input = dict()
        # todo: refactor all computed args to private variables?
        self._regression_penalty = None
        self._with_mcmc = 0
        # todo: should this be based on number of obs?
        self.min_nu = 5.
        self.max_nu = 40.
        # positive regressors
        self.num_of_positive_regressors = 0
        self.positive_regressor_col = list()
        self.positive_regressor_beta_prior = list()
        self.positive_regressor_sigma_prior = list()
        # regular regressors
        self.num_of_regular_regressors = 0
        self.regular_regressor_col = list()
        self.regular_regressor_beta_prior = list()
        self.regular_regressor_sigma_prior = list()
        
        # set static data attributes
        self._set_static_data_attributes()

        # set model param names
        # this only depends on static attributes, but should these params depend
        # on dynamic data (e.g actual data matrix, number of responses, etc) this should be
        # called after fit instead
        self._set_model_param_names()
        
        # init dynamic data attributes
        # the following are set by `_set_dynamic_data_attributes()` and generally set during fit()
        # from input df
        # response data
        self.response = None
        self.num_of_observations = None
        self.cauchy_sd = None
        self.stan_init = 'random'
        # regression data
        self.positive_regressor_matrix = None
        self.regular_regressor_matrix = None

        # init posterior samples
        # `_posterior_samples` is set by `fit()`
        self._posterior_samples = dict()
        
    def _set_regression_penalty(self):
        regression_penalty = self.regression_penalty
        self._regression_penalty = getattr(lgt.RegressionPenalty, regression_penalty).value

    def _set_static_regression_attributes(self):
        def _validate(regression_params, valid_length):
            for p in regression_params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

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

    def _set_with_mcmc(self):
        estimator_type = self.estimator_type
        # set `_with_mcmc` attribute based on estimator type
        # if no attribute for _is_mcmc_estimator, default to False
        if getattr(estimator_type, '_is_mcmc_estimator', False):
            self._with_mcmc = 1

    def _set_static_data_attributes(self):
        """Stan data input based on args at instatiation or computed from args at instantiation"""
        self._set_regression_penalty()
        self._set_static_regression_attributes()
        self._set_with_mcmc()

    def _set_training_df_meta(self, df):
        # Date Metadata
        # TODO: use from constants for dict key
        self._training_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'training_start': df[self.date_col].iloc[0],
            'training_end': df[self.date_col].iloc[-1]
        }

    def _validate_training_df(self, df):
        df_columns = df.columns

        # validate date_col
        if self.date_col not in df_columns:
            raise LGTException("DataFrame does not contain `date_col`: {}".format(self.date_col))

        # validate ordering of time series
        date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        if not is_ordered_datetime(date_array):
            raise LGTException('Datetime index must be ordered and not repeat')

        # validate regression columns
        if self.regressor_col is not None and \
                not set(self.regressor_col).issubset(df_columns):
            raise LGTException(
                "DataFrame does not contain specified regressor colummn(s)."
            )

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise LGTException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def _set_regressor_matrix(self):
        # init of regression matrix depends on length of response vector
        self.positive_regressor_matrix = np.zeros((self.num_of_observations, 0), dtype=np.double)
        self.regular_regressor_matrix = np.zeros((self.num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self.num_of_positive_regressors > 0:
            self.positive_regressor_matrix = self.df.filter(
                items=self.positive_regressor_col,).values

        if self.num_of_regular_regressors > 0:
            self.regular_regressor_matrix = self.df.filter(
                items=self.regular_regressor_col,).values

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

    def _set_stan_init(self):
        # # to use stan default, set self.stan_int to 'random'
        # self.stan_init = []
        # # use the seed so we can replicate results with same seed
        # np.random.seed(self.seed)
        # # ch is not used but we need the for loop to append init points across chains
        # for ch in range(self.chains):
        #     temp_init = {}
        #     if self.seasonality > 1:
        #         # note that although seed fixed, points are different across chains
        #         seas_init = np.random.normal(loc=0, scale=0.05, size=self.seasonality - 1)
        #         seas_init[seas_init > 1.0] = 1.0
        #         seas_init[seas_init < -1.0] = -1.0
        #         temp_init['init_sea'] = seas_init
        #     self.stan_init.append(temp_init)
        # todo: logic to pass a function that is not chain dependent
        #   and init logic will occur in the estimator not model
        pass

    def _set_dynamic_data_attributes(self, df):
        """Stan data input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()

        self._validate_training_df(df)
        self._set_training_df_meta(df)

        if self.is_multiplicative:
            df = self._log_transform_df(df, do_fit=True)

        # a few of the following are related with training data.
        self.response = df[self.response_col].values
        self.num_of_observations = len(self.response)

        self.cauchy_sd = max(self.response) / 30.0

        self._set_regressor_matrix()  # depends on num_of_observations
        self._set_stan_init()
        # TODO: maybe useful to calculate vanlia (adjusted) R-Squared
        # self._adjust_smoothing_with_regression()

    def _set_model_param_names(self):
        """Model parameters to extract from Stan"""
        self._model_param_names += [param.value for param in lgt.BaseSamplingParameters]

        # append seasonality param names
        if self.seasonality > 1:
            self._model_param_names += [param.value for param in lgt.SeasonalitySamplingParameters]

        # append positive regressors if any
        if self.num_of_positive_regressors > 0:
            self._model_param_names += [
                lgt.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value]

        # append regular regressors if any
        if self.num_of_regular_regressors > 0:
            self._model_param_names += [
                lgt.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value]

    def _get_model_param_names(self):
        return self._model_param_names

    def _set_stan_data_input(self):
        """Collects data attributes into a dict for `StanModel.sampling`"""
        data_inputs = dict()

        for key in self._data_input_mapper:
            # mapper keys in upper case; inputs in lower case
            key_lower = key.name.lower()
            input_value = getattr(self, key_lower, None)
            if input_value is None:
                raise LGTException('{} is missing from stan input'.format(key_lower))
            if isinstance(input_value, bool):
                # stan accepts bool as int only
                input_value = int(input_value)
            data_inputs[key.value] = input_value

        self._stan_data_input = data_inputs

    def _get_stan_data_input(self):
        return self._stan_data_input

    def _get_stan_init(self):
        return self.stan_init

    def is_fitted(self):
        # if empty dict false, else true
        return bool(self._posterior_samples)

    @staticmethod
    def _concat_regression_coefs(pr_beta=None, rr_beta=None):
        """Concatenates regression posterior matrix

        In the case that `pr_beta` or `rr_beta` is a 1d tensor, transform to 2d tensor and
        concatenate.

        Args
        ----
        pr_beta : torch.tensor
            postive-value constrainted regression betas
        rr_beta : torch.tensor
            regular regression betas

        Returns
        -------
        torch.tensor
            concatenated 2d tensor of shape (1, len(rr_beta) + len(pr_beta))

        """
        regressor_beta = None
        if pr_beta is not None and rr_beta is not None:
            pr_beta = pr_beta if len(pr_beta.shape) == 2 else pr_beta.reshape(1, -1)
            rr_beta = rr_beta if len(rr_beta.shape) == 2 else rr_beta.reshape(1, -1)
            regressor_beta = torch.cat((pr_beta, rr_beta), dim=1)
        elif pr_beta is not None:
            regressor_beta = pr_beta
        elif rr_beta is not None:
            regressor_beta = rr_beta

        return regressor_beta

    def _predict(self, posterior_estimates, df, include_error=False, decompose=False):
        """Vectorized version of prediction math"""

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
            lgt.SeasonalitySamplingParameters.SEASONALITY_LEVELS.value)
        seasonality_smoothing_factor = model.get(
            lgt.SeasonalitySamplingParameters.SEASONALITY_SMOOTHING_FACTOR.value
        )

        # trend components
        slope_smoothing_factor = model.get(
            lgt.BaseSamplingParameters.SLOPE_SMOOTHING_FACTOR.value)
        level_smoothing_factor = model.get(
            lgt.BaseSamplingParameters.LEVEL_SMOOTHING_FACTOR.value)
        local_trend_levels = model.get(lgt.BaseSamplingParameters.LOCAL_TREND_LEVELS.value)
        local_trend_slopes = model.get(lgt.BaseSamplingParameters.LOCAL_TREND_SLOPES.value)
        residual_degree_of_freedom = model.get(
            lgt.BaseSamplingParameters.RESIDUAL_DEGREE_OF_FREEDOM.value)
        residual_sigma = model.get(lgt.BaseSamplingParameters.RESIDUAL_SIGMA.value)

        local_trend_coef = model.get(lgt.BaseSamplingParameters.LOCAL_TREND_COEF.value)
        global_trend_power = model.get(lgt.BaseSamplingParameters.GLOBAL_TREND_POWER.value)
        global_trend_coef = model.get(lgt.BaseSamplingParameters.GLOBAL_TREND_COEF.value)
        local_global_trend_sums = model.get(
            lgt.BaseSamplingParameters.LOCAL_GLOBAL_TREND_SUMS.value)

        # regression components
        pr_beta = model.get(lgt.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value)
        rr_beta = model.get(lgt.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value)
        regressor_beta = self._concat_regression_coefs(pr_beta, rr_beta)

        ################################################################
        # Prediction Attributes
        ################################################################

        # get training df meta
        training_df_meta = self._training_df_meta
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
                trend_component += error_value
        else:
            trend_component = local_global_trend_sums
            # in-sample error are iids
            if include_error:
                error_value = nct.rvs(
                    df=residual_degree_of_freedom.unsqueeze(-1),
                    nc=0,
                    loc=0,
                    scale=residual_sigma.unsqueeze(-1),
                    size=(num_sample, local_global_trend_sums.shape[1])
                )

                error_value = torch.from_numpy(
                    error_value.reshape(num_sample, local_global_trend_sums.shape[1])).double()
                trend_component += error_value

            trend_forecast_matrix = torch.zeros((num_sample, n_forecast_steps), dtype=torch.double)
            trend_component = torch.cat((trend_component, trend_forecast_matrix), dim=1)

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

    def _prepend_date_column(self, predicted_df, input_df):
        """Prepends date column from `input_df` to `predicted_df`"""

        other_cols = list(predicted_df.columns)

        # add date column
        predicted_df[self.date_col] = input_df[self.date_col].reset_index(drop=True)

        # re-order columns so date is first
        col_order = [self.date_col] + other_cols
        predicted_df = predicted_df[col_order]

        return predicted_df


class LGTFull(BaseLGT):
    def __init__(self, n_bootstrap_draws=-1, prediction_percentiles=None, **kwargs):
        # todo: assert compatible estimator
        super().__init__(**kwargs)
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles

        # set default args
        self._prediction_percentiles = None
        self._set_default_args()

    def _set_default_args(self):
        if not self.prediction_percentiles:
            self._prediction_percentiles = list()
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

    def _bootstrap(self, n):
        """Draw `n` number of bootstrap samples from the posterior_samples.

        Args
        ----
        n : int
            The number of bootstrap samples to draw

        """
        num_samples = self.estimator.num_sample
        posterior_samples = self._posterior_samples

        if n < 2:
            raise IllegalArgument("Error: The number of bootstrap draws must be at least 2")

        sample_idx = np.random.choice(
            range(num_samples),
            size=n,
            replace=True
        )

        bootstrap_samples_dict = {}
        for k, v in posterior_samples.items():
            bootstrap_samples_dict[k] = v[sample_idx]

        return bootstrap_samples_dict

    def _aggregate_full_predictions(self, predictions_array):
        """Aggregates the mcmc prediction to a point estimate

        Args
        ----
        predictions_array : np.ndarray
            A 2d numpy array of shape (`num_samples`, prediction df length)
        percentiles : list
            The percentiles at which to aggregate the predictions

        Returns
        -------
        pd.DataFrame
            The aggregated across mcmc samples with columns for `mean`, `50` aka median
            and all other percentiles specified in `percentiles`.

        """

        # MUST copy, or else instance var persists in memory
        percentiles = copy(self._prediction_percentiles)

        percentiles += [50]  # always find median
        percentiles.sort()

        # mean_prediction = np.mean(predictions_array, axis=0)
        percentiles_prediction = np.percentile(predictions_array, percentiles, axis=0)

        aggregate_df = pd.DataFrame(percentiles_prediction.T, columns=percentiles)

        # rename `50` to `prediction`
        aggregate_df.rename(columns={50: 'prediction'}, inplace=True)

        return aggregate_df

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        estimator = self.estimator
        stan_model_name = self._stan_model_name

        self._set_dynamic_data_attributes(df)
        self._set_stan_data_input()

        # estimator inputs
        data_input = self._get_stan_data_input()
        stan_init = self._get_stan_init()
        model_param_names = self._get_model_param_names()

        stan_extract = estimator.fit(
            stan_model_name=stan_model_name,
            model_param_names=model_param_names,
            data_input=data_input,
            stan_init=stan_init
        )

        self._posterior_samples = stan_extract

    def predict(self, df):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(self.n_bootstrap_draws) \
            if self.n_bootstrap_draws > 1 \
            else self._posterior_samples

        predictions_array = self._predict(
            posterior_estimates=posterior_samples,
            df=df,
            include_error=True
        )['prediction']

        aggregated_df = self._aggregate_full_predictions(predictions_array)

        aggregated_df = self._prepend_date_column(aggregated_df, df)

        return aggregated_df


class LGTAggregated(object):
    def __init__(self, estimator_class, model_params, **kwargs):
        pass


class LGTMAP(object):
    def __init__(self, estimator_class, model_params, **kwargs):
        pass