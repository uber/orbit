import pandas as pd
import numpy as np
import math
from scipy.stats import nct
import torch
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..constants import gam as constants
from ..constants.constants import (
    # COEFFICIENT_DF_COLS,
    PredictMethod
)
from ..constants.gam import (
    DEFAULT_LEVEL_KNOT_SCALE,
    DEFAULT_PR_KNOT_POOL_SCALE,
    DEFAULT_SPAN_LEVEL,
    DEFAULT_SPAN_COEFFICIENTS,
    DEFAULT_RHO_LEVEL,
    DEFAULT_RHO_COEFFICIENTS,
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_COEFFICIENTS_LOC,
    DEFAULT_COEFFICIENTS_SCALE,
    DEFAULT_DEGREE_OF_FREEDOM,
)

from ..estimators.pyro_estimator import PyroEstimatorVI, PyroEstimatorMAP
from ..exceptions import IllegalArgument, ModelException, PredictionException
from .base_model import BaseModel
from ..utils.general import is_ordered_datetime
from ..utils.kernels import gauss_kernel


class BaseGAM(BaseModel):
    """Base LGT model object with shared functionality for Full, Aggregated, and MAP methods

    Parameters
    ----------
    response_col : str
        Name of response variable column, default 'y'
    date_col : str
        Name of date variable column, default 'ds'
    regressor_col : list
        Names of regressor columns, if any
    seasonality : int
        Length of seasonality
    regressor_sign :  list
        list with values { '+', '=' }. '+' indicates regressor coefficient estimates are
        constrained to [0, inf). '=' indicates regressor coefficient estimates
        can be any value between (-inf, inf). The length of `regressor_sign` must be
        the same length as `regressor_col`. If None, all elements of list will be set
        to '='.
    regressor_beta_prior : list
        list of prior float values for regressor coefficient betas. The length of `regressor_beta_prior`
        must be the same length as `regressor_col`. If None, use non-informative priors.
    regressor_sigma_prior : list
        list of prior float values for regressor coefficient sigmas. The length of `regressor_sigma_prior`
        must be the same length as `regressor_col`. If None, use non-informative priors.
    kwargs
        To specify `estimator_type` or additional args for the specified `estimator_type`

    """
    _data_input_mapper = constants.DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'gam'
    _supported_estimator_types = None  # set for each model

    def __init__(self,
                 response_col='y',
                 date_col='ds',
                 level_knot_scale=None,
                 regressor_col=None,
                 regressor_sign=None,
                 regressor_knot_loc=None,
                 regressor_knot_scale=None,
                 positive_regressor_knot_pooling_scale=None,
                 span_level=None,
                 span_coefficients=None,
                 rho_level=None,
                 rho_coefficients=None,
                 degree_of_freedom=30,
                 # response_sd=None,
                 insert_prior_idx=None,
                 insert_prior_tp_idx=None,
                 insert_prior_mean=None,
                 insert_prior_sd=None,
                 **kwargs):
        super().__init__(**kwargs)  # create estimator in base class
        self.response_col = response_col
        self.date_col = date_col
        self.level_knot_scale = level_knot_scale
        self.regressor_col = regressor_col
        self.regressor_sign = regressor_sign
        self.regressor_knot_loc = regressor_knot_loc
        self.regressor_knot_scale = regressor_knot_scale
        self.positive_regressor_knot_pooling_scale = positive_regressor_knot_pooling_scale
        self.span_level = span_level
        self.span_coefficients = span_coefficients
        self.rho_level = rho_level
        self.rho_coefficients = rho_coefficients
        self.insert_prior_idx = insert_prior_idx
        self.insert_prior_tp_idx = insert_prior_tp_idx
        self.insert_prior_mean = insert_prior_mean
        self.insert_prior_sd = insert_prior_sd

        # set private var to arg value
        # if None set default in _set_default_base_args()
        self._level_knot_scale = self.level_knot_scale
        self._regressor_sign = self.regressor_sign
        self._regressor_knot_loc = self.regressor_knot_loc
        self._regressor_knot_scale = self.regressor_knot_scale
        self._span_level = self.span_level
        self._span_coefficients = self.span_coefficients
        self._rho_level = self.rho_level
        self._rho_coefficients = self.rho_coefficients
        self._positive_regressor_knot_pooling_scale = self.positive_regressor_knot_pooling_scale
        self._degree_of_freedom = degree_of_freedom
        self._insert_prior_idx = self.insert_prior_idx
        self._insert_prior_tp_idx = self.insert_prior_tp_idx
        self._insert_prior_mean = self.insert_prior_mean
        self._insert_prior_sd = self.insert_prior_sd

        self._model_param_names = list()
        self._training_df_meta = None
        self._model_data_input = dict()

        # positive regressors
        self._num_of_positive_regressors = 0
        self._positive_regressor_col = list()
        self._positive_regressor_knot_pooling_loc = list()
        # self._positive_regressor_knot_pooling_scale = list()
        self._positive_regressor_knot_scale = list()
        # self._positive_regressor_latent_sale_prior = list()
        # regular regressors
        self._num_of_regular_regressors = 0
        self._regular_regressor_col = list()
        self._regular_regressor_knot_loc = list()
        self._regular_regressor_knot_scale = list()

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
        self._response = None
        self._num_of_observations = None
        self._response_sd = None
        self._num_insert_prior = None
        self._num_knots_level = None
        self._num_knots_coefficients = None
        self._knots_tp_level = None
        self.n_knots_coefficients = None
        # regression data
        self._positive_regressor_matrix = None
        self._regular_regressor_matrix = None

        # init posterior samples
        # `_posterior_samples` is set by `fit()`
        self._posterior_samples = dict()

        # init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }

    def _set_default_base_args(self):
        """Set default attributes for None

        Stan requires static data types so data must be cast to the correct type
        """

        if self.level_knot_scale is None:
            self._level_knot_scale = DEFAULT_LEVEL_KNOT_SCALE
        if self.positive_regressor_knot_pooling_scale is None:
            self._positive_regressor_knot_pooling_scale = DEFAULT_PR_KNOT_POOL_SCALE
        if self.span_level is None:
            self._span_level = DEFAULT_SPAN_LEVEL
        if self.span_coefficients is None:
            self._span_coefficients = DEFAULT_SPAN_COEFFICIENTS
        if self.rho_level is None:
            self._rho_level = DEFAULT_RHO_LEVEL
        if self.rho_coefficients is None:
            self._rho_coefficients = DEFAULT_RHO_COEFFICIENTS

        if self.insert_prior_idx is None:
            self._insert_prior_idx = list()
        if self.insert_prior_tp_idx is None:
            self._insert_prior_tp_idx = list()
        if self.insert_prior_mean is None:
            self._insert_prior_mean = list()
        if self.insert_prior_sd is None:
            self._insert_prior_sd = list()

        ##############################
        # if no regressors, end here #
        ##############################
        if self.regressor_col is None:
            # regardless of what args are set for these, if regressor_col is None
            # these should all be empty lists
            self._regressor_sign = list()
            self._regressor_knot_loc = list()
            self._regressor_knot_scale = list()

            return

        def _validate(regression_params, valid_length):
            for p in regression_params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

        def _validate_insert_prior(insert_prior_params):
            len_insert_prior = list()
            for p in insert_prior_params:
                if p is not None:
                    len_insert_prior.append(len(p))
                else: len_insert_prior.append(0)
            if not all(len_insert == len_insert_prior[0] for len_insert in len_insert_prior):
                raise IllegalArgument('Wrong dimension length in Insert Prior Input')

        # regressor defaults
        num_of_regressors = len(self.regressor_col)

        _validate(
            [self.regressor_sign, self.regressor_knot_loc, self.regressor_knot_scale],
            num_of_regressors
        )
        _validate_insert_prior([self.insert_prior_idx, self.insert_prior_tp_idx,
                                self.insert_prior_mean, self.insert_prior_sd])

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * num_of_regressors

        if self.regressor_knot_loc is None:
            self._regressor_knot_loc = [DEFAULT_COEFFICIENTS_LOC] * num_of_regressors

        if self.regressor_knot_scale is None:
            self._regressor_knot_scale = [DEFAULT_COEFFICIENTS_SCALE] * num_of_regressors

    def _set_static_regression_attributes(self):
        # if no regressors, end here
        if self.regressor_col is None:
            return

        # inside *.stan files, we need to distinguish regular regressors from positive regressors
        for index, reg_sign in enumerate(self._regressor_sign):
            if reg_sign == '+':
                self._num_of_positive_regressors += 1
                self._positive_regressor_col.append(self.regressor_col[index])
                # used for 'pr_knot_loc' sampling in pyro
                self._positive_regressor_knot_pooling_loc.append(self._regressor_knot_loc[index])
                # used for 'pr_knot' sampling in pyro
                self._positive_regressor_knot_scale.append(self._regressor_knot_scale[index])
            else:
                self._num_of_regular_regressors += 1
                self._regular_regressor_col.append(self.regressor_col[index])
                self._regular_regressor_knot_loc.append(self._regressor_knot_loc[index])
                self._regular_regressor_knot_scale.append(self._regressor_knot_scale[index])

    def _set_static_data_attributes(self):
        """model data input based on args at instatiation or computed from args at instantiation"""
        self._set_default_base_args()
        self._set_static_regression_attributes()

    def _validate_supported_estimator_type(self):
        if self.estimator_type not in self._supported_estimator_types:
            msg_template = "Model class: {} is incompatible with Estimator: {}"
            model_class = type(self)
            estimator_type = self.estimator_type
            raise IllegalArgument(msg_template.format(model_class, estimator_type))

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
            raise ModelException("DataFrame does not contain `date_col`: {}".format(self.date_col))

        # validate ordering of time series
        date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        if not is_ordered_datetime(date_array):
            raise ModelException('Datetime index must be ordered and not repeat')

        # validate regression columns
        if self.regressor_col is not None and \
                not set(self.regressor_col).issubset(df_columns):
            raise ModelException(
                "DataFrame does not contain specified regressor colummn(s)."
            )

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ModelException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def _set_regressor_matrix(self, df):
        # init of regression matrix depends on length of response vector
        self._positive_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)
        self._regular_regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self._num_of_positive_regressors > 0:
            self._positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col,).values

        if self._num_of_regular_regressors > 0:
            self._regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col,).values

    def _set_kernel_matrix(self):
        tp = np.arange(1, self._num_of_observations + 1) / self._num_of_observations
        # this approach put knots in full range
        self._cutoff = self._num_of_observations
        # cutoff last 20%
        # self._cutoff = round(0.2 * self._num_of_observations)
        width_level = round(self._span_level * self._cutoff)
        width_coefficients = round(self._span_coefficients * self._cutoff)
        self._knots_tp_level = np.arange(1, self._cutoff + 1, width_level) / self._num_of_observations
        self.n_knots_coefficients = np.arange(1, self._cutoff + 1, width_coefficients) / self._num_of_observations

        # Kernel here is used to determine mean
        kernel_level = gauss_kernel(tp, self._knots_tp_level, rho=self._rho_level)
        kernel_coefficients = gauss_kernel(tp, self.n_knots_coefficients, rho=self._rho_coefficients)
        kernel_level = kernel_level/np.sum(kernel_level, axis=1, keepdims=True)
        kernel_coefficients = kernel_coefficients / np.sum(kernel_coefficients, axis=1, keepdims=True)

        self._num_knots_level = len(self._knots_tp_level)
        self._num_knots_coefficients = len(self.n_knots_coefficients)
        self._kernel_level = kernel_level
        self._kernel_coefficients = kernel_coefficients

    def _set_dynamic_data_attributes(self, df):
        """Stan data input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()

        self._validate_training_df(df)
        self._set_training_df_meta(df)

        # a few of the following are related with training data.
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)
        self._response_sd = np.std(self._response)
        self._num_insert_prior = len(self._insert_prior_mean)

        self._set_regressor_matrix(df)
        self._set_kernel_matrix()

    def _set_model_param_names(self):
        """Model parameters to extract from Stan"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]
        self._model_param_names += [param.value for param in constants.RegressionSamplingParameters]

        # # append seasonality param names
        # if self._seasonality > 1:
        #     self._model_param_names += [param.value for param in constants.SeasonalitySamplingParameters]

        # # append positive regressors if any
        # if self._num_of_positive_regressors > 0:
        #     self._model_param_names += [
        #         constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value]

        # # append regular regressors if any
        # if self._num_of_regular_regressors > 0:
        #     self._model_param_names += [
        #         constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value]

    def _get_model_param_names(self):
        return self._model_param_names

    def _set_model_data_input(self):
        """Collects data attributes into a dict for `StanModel.sampling`"""
        data_inputs = dict()

        for key in self._data_input_mapper:
            # mapper keys in upper case; inputs in lower case
            key_lower = key.name.lower()
            input_value = getattr(self, key_lower, None)
            if input_value is None:
                raise ModelException('{} is missing from data input'.format(key_lower))
            if isinstance(input_value, bool):
                # stan accepts bool as int only
                input_value = int(input_value)
            data_inputs[key.value] = input_value

        self._model_data_input = data_inputs

    def _get_model_data_input(self):
        return self._model_data_input

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
            regressor_beta = torch.cat((rr_beta, pr_beta), dim=1)
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
        # for k, v in model.items():
        #     model[k] = torch.from_numpy(v)

        # We can pull any arbitrary value from the dictionary because we hold the
        # safe assumption: the length of the first dimension is always the number of samples
        # thus can be safely used to determine `num_sample`. If predict_method is anything
        # other than full, the value here should be 1
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

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

        trained_len = training_df_meta['df_length'] # i.e., self._num_of_observations
        output_len = prediction_df_meta['df_length']

        # Here assume dates are ordered and consecutive
        # if prediction_df_meta['prediction_start'] > training_df_meta['training_end'],
        # assume prediction starts right after train end
        # TODO: check utility?
        gap_time = prediction_df_meta['prediction_start'] - training_df_meta['training_start']
        infer_freq = pd.infer_freq(df[self.date_col])[0]
        gap_int = int(gap_time / np.timedelta64(1, infer_freq))

        new_tp = np.arange(1 + gap_int, output_len + gap_int + 1)
        new_tp = new_tp / trained_len
        kernel_level = gauss_kernel(new_tp, self._knots_tp_level, rho=self._rho_level)
        kernel_coefficients = gauss_kernel(new_tp, self.n_knots_coefficients, rho=self._rho_coefficients)
        kernel_level = kernel_level/np.sum(kernel_level, axis=1, keepdims=True)
        kernel_coefficients = kernel_coefficients / np.sum(kernel_coefficients, axis=1, keepdims=True)

        lev_knot = model.get(constants.BaseSamplingParameters.LEVEL_KNOT.value)
        coef_knot = model.get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
        obs_scale = model.get(constants.BaseSamplingParameters.OBS_SCALE.value)
        obs_scale = obs_scale.reshape(-1, 1)

        # init of regression matrix depends on length of response vector
        pred_positive_regressor_matrix = np.zeros((output_len, 0), dtype=np.double)
        pred_regular_regressor_matrix = np.zeros((output_len, 0), dtype=np.double)
        # update regression matrices
        if self._num_of_positive_regressors > 0:
            pred_positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col,).values
        if self._num_of_regular_regressors > 0:
            pred_regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col,).values
        # regular first, then positive
        pred_regressor_matrix = np.concatenate([pred_regular_regressor_matrix,
                                                pred_positive_regressor_matrix], axis=-1)

        trend = np.matmul(lev_knot, kernel_level.transpose(1, 0))
        regression = np.sum(np.matmul(coef_knot, kernel_coefficients.transpose(1, 0)) * \
                            pred_regressor_matrix.transpose(1, 0), axis=-2)
        if include_error:
            epsilon = nct.rvs(self._degree_of_freedom, nc=0, loc=0, scale=obs_scale, size=(num_sample, len(new_tp)))
            pred_array = trend + regression + epsilon
        else:
            pred_array = trend + regression

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend,
                'regression': regression
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

    def _set_aggregate_posteriors(self):
        posterior_samples = self._posterior_samples

        mean_posteriors = {}
        median_posteriors = {}

        # for each model param, aggregate using `method`
        for param_name in self._model_param_names:
            param_ndarray = posterior_samples[param_name]

            mean_posteriors.update(
                {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
            )

            median_posteriors.update(
                {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
            )

        self._aggregate_posteriors[PredictMethod.MEAN.value] = mean_posteriors
        self._aggregate_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        estimator = self.estimator
        model_name = self._model_name

        self._set_dynamic_data_attributes(df)
        self._set_model_data_input()

        # estimator inputs
        data_input = self._get_model_data_input()
        # init_values = self._get_init_values()
        model_param_names = self._get_model_param_names()

        model_extract = estimator.fit(
            model_name=model_name,
            model_param_names=model_param_names,
            data_input=data_input,
            # init_values=init_values
        )

        self._posterior_samples = model_extract

    def get_regression_coefs(self, aggregate_method, include_ci=False):
        """Return DataFrame regression coefficients

        If PredictMethod is `full` return `mean` of coefficients instead
        """
        # init dataframe
        reg_df = pd.DataFrame()
        reg_df[self.date_col] = self._training_df_meta['date_array']

        # end if no regressors
        if self._num_of_regular_regressors + self._num_of_positive_regressors == 0:
            return reg_df

        regressor_betas = self._aggregate_posteriors \
            .get(aggregate_method) \
            .get(constants.RegressionSamplingParameters.COEFFICIENTS.value)
        regressor_betas = np.squeeze(regressor_betas, axis=0)

        # pr_beta = self._aggregate_posteriors\
        #     .get(aggregate_method)\
        #     .get(constants.RegressionSamplingParameters.POSITIVE_REGRESSOR_BETA.value)

        # rr_beta = self._aggregate_posteriors\
        #     .get(aggregate_method)\
        #     .get(constants.RegressionSamplingParameters.REGULAR_REGRESSOR_BETA.value)

        # # because `_conccat_regression_coefs` operates on torch tensors
        # pr_beta = torch.from_numpy(pr_beta) if pr_beta is not None else pr_beta
        # rr_beta = torch.from_numpy(rr_beta) if rr_beta is not None else rr_beta

        # # regular first, then positive
        # regressor_betas = self._concat_regression_coefs(rr_beta, pr_beta)

        # get column names
        pr_cols = self._positive_regressor_col
        rr_cols = self._regular_regressor_col

        # note ordering here is not the same as `self.regressor_cols` because positive
        # and negative do not have to be grouped on input
        regressor_col = rr_cols + pr_cols
        for idx, col in enumerate(regressor_col):
            reg_df[col] = regressor_betas[:, idx]

        if include_ci:
            posterior_samples = self._posterior_samples
            param_ndarray = posterior_samples.get(constants.RegressionSamplingParameters.COEFFICIENTS.value)
            coefficients_lower = np.quantile(param_ndarray, [0.05], axis=0)
            coefficients_upper = np.quantile(param_ndarray, [0.95], axis=0)
            coefficients_lower = np.squeeze(coefficients_lower, axis=0)
            coefficients_upper = np.squeeze(coefficients_upper, axis=0)

            reg_df_lower = reg_df.copy()
            reg_df_upper = reg_df.copy()
            for idx, col in enumerate(regressor_col):
                reg_df_lower[col] = coefficients_lower[:, idx]
                reg_df_upper[col] = coefficients_upper[:, idx]
            return reg_df, reg_df_lower, reg_df_upper

        return reg_df

    def plot_regression_coefs(self,
                              coef_df,
                              coef_df_lower=None,
                              coef_df_upper=None,
                              ncol=2,
                              figsize=None,
                              ylim=None):
        nrow = math.ceil((coef_df.shape[1] - 1) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        regressor_col = coef_df.columns.tolist()[1:]

        for idx, col in enumerate(regressor_col):
            row_idx = idx // ncol
            col_idx = idx % ncol
            coef = coef_df[col]
            axes[row_idx, col_idx].plot(coef, alpha=.8)  # label?
            if coef_df_lower is not None and coef_df_upper is not None:
                coef_lower = coef_df_lower[col]
                coef_upper = coef_df_upper[col]
                axes[row_idx, col_idx].fill_between(np.arange(0, coef_df.shape[0]), coef_lower, coef_upper, alpha=.3)
            if ylim is not None: axes[row_idx, col_idx].set_ylim(ylim)
            # regressor_col_names = ['intercept'] + self.regressor_col if self.intercept else self.regressor_col
            axes[row_idx, col_idx].set_title('{}'.format(col))
            axes[row_idx, col_idx].ticklabel_format(useOffset=False)
        plt.tight_layout()

        return fig


class GAMFull(BaseGAM):
    """Concrete LGT model for full prediction

    In full prediction, the prediction occurs as a function of each parameter posterior sample,
    and the prediction results are aggregated after prediction. Prediction will
    always return the median (aka 50th percentile) along with any additional percentiles that
    are specified.

    Parameters
    ----------
    n_bootstrap_draws : int
        Number of bootstrap samples to draw from the initial MCMC or VI posterior samples.
        If None, use the original posterior draws.
    prediction_percentiles : list
        List of integers of prediction percentiles that should be returned on prediction. To avoid reporting any
        confident intervals, pass an empty list
    kwargs
        Additional args to pass to parent classes.

    """
    _supported_estimator_types = [PyroEstimatorVI]

    def __init__(self, n_bootstrap_draws=None, prediction_percentiles=None, **kwargs):
        # todo: assert compatible estimator
        super().__init__(**kwargs)
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles

        # set default args
        self._prediction_percentiles = None
        self._n_bootstrap_draws = self.n_bootstrap_draws
        self._set_default_args()

        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _set_default_args(self):
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        if not self.n_bootstrap_draws:
            self._n_bootstrap_draws = -1

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

    def _aggregate_full_predictions(self, array, label, percentiles):
        """Aggregates the mcmc prediction to a point estimate
        Args
        ----
        array: np.ndarray
            A 2d numpy array of shape (`num_samples`, prediction df length)
        percentiles: list
            A sorted list of one or three percentile(s) which will be used to aggregate lower, mid and upper values
        label: str
            A string used for labeling output dataframe columns
        Returns
        -------
        pd.DataFrame
            The aggregated across mcmc samples with columns for `50` aka median
            and all other percentiles specified in `percentiles`.
        """

        aggregated_array = np.percentile(array, percentiles, axis=0)
        columns = [label + "_" + str(p) if p != 50 else label for p in percentiles]
        aggregate_df = pd.DataFrame(aggregated_array.T, columns=columns)
        return aggregate_df

    def predict(self, df, decompose=False):
        """Return model predictions as a function of fitted model and df"""
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(self._n_bootstrap_draws) \
            if self._n_bootstrap_draws > 1 \
            else self._posterior_samples

        predicted_dict = self._predict(
            posterior_estimates=posterior_samples,
            df=df,
            include_error=True,
            decompose=decompose,
        )

        # MUST copy, or else instance var persists in memory
        percentiles = copy(self._prediction_percentiles)
        percentiles += [50]  # always find median
        percentiles = list(set(percentiles))  # unique set
        percentiles.sort()

        for k, v in predicted_dict.items():
            predicted_dict[k] = self._aggregate_full_predictions(
                array=v,
                label=k,
                percentiles=percentiles,
            )

        aggregated_df = pd.concat(predicted_dict, axis=1)
        aggregated_df.columns = aggregated_df.columns.droplevel()
        aggregated_df = self._prepend_date_column(aggregated_df, df)
        return aggregated_df

    def get_regression_coefs(self, aggregate_method='mean', include_ci=False):
        self._set_aggregate_posteriors()
        return super().get_regression_coefs(aggregate_method=aggregate_method,
                                            include_ci=include_ci)

    def plot_regression_coefs(self, aggregate_method='mean', include_ci=False, **kwargs):
        if include_ci:
            coef_df, coef_df_lower, coef_df_upper = self.get_regression_coefs(
                aggregate_method=aggregate_method,
                include_ci=True
            )
        else:
            coef_df = self.get_regression_coefs(aggregate_method=aggregate_method, include_ci=False)
            coef_df_lower = None
            coef_df_upper = None

        return super().plot_regression_coefs(coef_df=coef_df,
                                             coef_df_lower=coef_df_lower,
                                             coef_df_upper=coef_df_upper,
                                             **kwargs)

class GAMAggregated(BaseGAM):
    """Concrete LGT model for aggregated posterior prediction
    In aggregated prediction, the parameter posterior samples are reduced using `aggregate_method`
    before performing a single prediction.
    Parameters
    ----------
    aggregate_method : { 'mean', 'median' }
        Method used to reduce parameter posterior samples
    """
    _supported_estimator_types = [PyroEstimatorVI]

    def __init__(self, aggregate_method='mean', **kwargs):
        super().__init__(**kwargs)
        self.aggregate_method = aggregate_method

        self._validate_aggregate_method()

        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _validate_aggregate_method(self):
        if self.aggregate_method not in list(self._aggregate_posteriors.keys()):
            raise PredictionException("No aggregate method defined for: `{}`".format(self.aggregate_method))

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        super().fit(df)
        self._set_aggregate_posteriors()

    def predict(self, df):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(self.aggregate_method)

        predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors,
            df=df,
            include_error=False,
        )

        # must flatten to convert to DataFrame
        for k, v in predicted_dict.items():
            predicted_dict[k] = v.flatten()

        predicted_df = pd.DataFrame(predicted_dict)
        predicted_df = self._prepend_date_column(predicted_df, df)

        return predicted_df

    def get_regression_coefs(self):
        return super().get_regression_coefs(aggregate_method=self.aggregate_method, include_ci=False)

    def plot_regression_coefs(self, **kwargs):
        coef_df = self.get_regression_coefs()
        return super().plot_regression_coefs(coef_df=coef_df,
                                             **kwargs)


class GAMMAP(BaseGAM):
    """Concrete LGT model for MAP (Maximum a Posteriori) prediction
    Similar to `LGTAggregated` but predition is based on Maximum a Posteriori (aka Mode)
    of the posterior.
    This model only supports MAP estimating `estimator_type`s
    """
    _supported_estimator_types = [PyroEstimatorMAP]

    def __init__(self, estimator_type=PyroEstimatorMAP, **kwargs):
        super().__init__(estimator_type=estimator_type, **kwargs)

        # override init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MAP.value: dict(),
        }

        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _set_map_posterior(self):
        posterior_samples = self._posterior_samples

        map_posterior = {}
        for param_name in self._model_param_names:
            param_array = posterior_samples[param_name]
            # add dimension so it works with vector math in `_predict`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update(
                {param_name: param_array}
            )

        self._aggregate_posteriors[PredictMethod.MAP.value] = map_posterior

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        super().fit(df)
        self._set_map_posterior()

    def predict(self, df):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(PredictMethod.MAP.value)

        predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors,
            df=df,
            include_error=False,
        )

        # must flatten to convert to DataFrame
        for k, v in predicted_dict.items():
            predicted_dict[k] = v.flatten()

        predicted_df = pd.DataFrame(predicted_dict)
        predicted_df = self._prepend_date_column(predicted_df, df)

        return predicted_df

    def get_regression_coefs(self):
        return super().get_regression_coefs(aggregate_method=PredictMethod.MAP.value)

    def plot_regression_coefs(self, **kwargs):
        coef_df = self.get_regression_coefs()
        return super().plot_regression_coefs(coef_df=coef_df,
                                             **kwargs)
