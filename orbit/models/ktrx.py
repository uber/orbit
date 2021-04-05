import pandas as pd
import numpy as np
import math
from scipy.stats import nct
import torch
from copy import copy, deepcopy
import matplotlib.pyplot as plt

from ..constants import ktrx as constants
from ..constants.constants import (
    PredictMethod,
    CoefPriorDictKeys,
)
from ..constants.ktrx import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_COEFFICIENTS_KNOT_POOL_SCALE,
    DEFAULT_COEFFICIENTS_KNOT_POOL_LOC,
    DEFAULT_COEFFICIENTS_KNOT_SCALE,
)

from ..estimators.pyro_estimator import PyroEstimatorVI
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..utils.general import is_ordered_datetime
from ..utils.kernels import gauss_kernel, sandwich_kernel
from ..utils.features import make_fourier_series_df
from .template import BaseTemplate, FullBayesianTemplate, AggregatedPosteriorTemplate


class BaseKTRX(BaseTemplate):
    """Base KTRX model object with shared functionality for PyroVI method

    Parameters
    ----------
    level_knot_scale : float
        sigma for level; default to be .1
    regressor_col : array-like strings
        regressor columns
    regressor_sign : list
        list of signs with '=' for regular regressor and '+' for positive regressor
    regressor_knot_pooling_loc : list
        list of regressor knot pooling mean priors, default to be 0's
    regressor_knot_pooling_scale : list
        list of regressor knot pooling sigma's to control the pooling strength towards the grand mean of regressors;
        default to be 1.
    regressor_knot_scale : list
        list of regressor knot sigma priors; default to be 0.1.
    span_coefficients : float between (0, 1)
        window width to decide the number of windows for the regression term
    rho_coefficients : float
        sigma in the Gaussian kernel for the regression term
    degree of freedom : int
        degree of freedom for error t-distribution
    coef_prior_list : list of dicts
        each dict in the list should have keys as
        'name', prior_start_tp_idx' (inclusive), 'prior_end_tp_idx' (not inclusive),
        'prior_mean', 'prior_sd', and 'prior_regressor_col'
    level_knot_dates : array like
        list of pre-specified dates for level knots
    level_knots : array like
        list of knot locations for level
        level_knot_dates and level_knots should be of the same length
    seasonal_knots_input : dict
         a dictionary for seasonality inputs with the following keys:
            '_seas_coef_knot_dates' : knot dates for seasonal regressors
            '_sea_coef_knot' : knot locations for sesonal regressors
            '_sea_rho' : rho value for seasonal regressors
            '_seasonality' : seasonality order
            '_seasonality_fs_order' : fourier series order for seasonality
    coefficients_knot_length : int
        the distance between every two knots for coefficients
    coefficients_knot_dates : array like
        a list of pre-specified knot dates for coefficients
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    kwargs
        To specify `estimator_type` or additional args for the specified `estimator_type`

    """
    _data_input_mapper = constants.DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'ktrx'

    def __init__(self,
                 level_knot_scale=0.1,
                 regressor_col=None,
                 regressor_sign=None,
                 regressor_knot_pooling_loc=None,
                 regressor_knot_pooling_scale=None,
                 regressor_knot_scale=None,
                 span_coefficients=0.3,
                 rho_coefficients=0.15,
                 degree_of_freedom=30,
                 # time-based coefficient priors
                 coef_prior_list=None,
                 # knot customization
                 level_knot_dates=None,
                 level_knots=None,
                 seasonal_knots_input=None,
                 coefficients_knot_length=None,
                 coefficients_knot_dates=None,
                 date_freq=None,
                 **kwargs):
        super().__init__(**kwargs)  # create estimator in base class

        # normal distribution of known knot
        self.level_knot_scale = level_knot_scale

        self.level_knot_dates = level_knot_dates
        self._level_knot_dates = level_knot_dates

        self.level_knots = level_knots
        # self._level_knots = level_knots

        self._kernel_level = None
        self._num_knots_level = None
        self.knots_tp_level = None

        self._seasonal_knots_input = seasonal_knots_input
        self._seas_term = 0

        self.regressor_col = regressor_col
        self.regressor_sign = regressor_sign
        self.regressor_knot_pooling_loc = regressor_knot_pooling_loc
        self.regressor_knot_pooling_scale = regressor_knot_pooling_scale
        self.regressor_knot_scale = regressor_knot_scale

        self.coefficients_knot_length = coefficients_knot_length
        self.span_coefficients = span_coefficients
        self.rho_coefficients = rho_coefficients
        self.date_freq = date_freq

        self.degree_of_freedom = degree_of_freedom

        # set private var to arg value
        # if None set default in _set_default_args()
        self._regressor_sign = self.regressor_sign
        self._regressor_knot_pooling_loc = self.regressor_knot_pooling_loc
        self._regressor_knot_pooling_scale= self.regressor_knot_pooling_scale
        self._regressor_knot_scale = self.regressor_knot_scale

        self.coef_prior_list = coef_prior_list
        self._coef_prior_list = []
        self._coefficients_knot_dates = coefficients_knot_dates

        self._num_of_regressors = 0
        self._num_knots_coefficients = 0

        # positive regressors
        self._num_of_positive_regressors = 0
        self._positive_regressor_col = list()
        self._positive_regressor_knot_pooling_loc = list()
        self._positive_regressor_knot_pooling_scale = list()
        self._positive_regressor_knot_scale = list()
        # regular regressors
        self._num_of_regular_regressors = 0
        self._regular_regressor_col = list()
        self._regular_regressor_knot_pooling_loc = list()
        self._regular_regressor_knot_pooling_scale = list()
        self._regular_regressor_knot_scale = list()
        self._regressor_col = list()

        # init dynamic data attributes
        # the following are set by `_set_dynamic_attributes()` and generally set during fit()
        # from input df
        # response data
        self._is_valid_response = None
        self._which_valid_response = None
        self._num_of_valid_response = 0
        self._seasonality = None

        # regression data
        self._knots_tp_coefficients = None
        self._positive_regressor_matrix = None
        self._regular_regressor_matrix = None

    def _set_model_param_names(self):
        """Overriding base template functions. Model parameters to extract"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]
        if self._num_of_regressors > 0 :
            self._model_param_names += [param.value for param in constants.RegressionSamplingParameters]

    def _set_default_args(self):
        """Set default attributes for None
        """
        if self.coef_prior_list is not None:
            self._coef_prior_list = deepcopy(self.coef_prior_list)
        if self.level_knots is None:
            self._level_knots = list()
        if self._seasonal_knots_input is not None:
            self._seasonality = self._seasonal_knots_input['_seasonality']
        else:
            self._seasonality = list()
        ##############################
        # if no regressors, end here #
        ##############################
        if self.regressor_col is None:
            # regardless of what args are set for these, if regressor_col is None
            # these should all be empty lists
            self._regressor_sign = list()
            self._regressor_knot_pooling_loc = list()
            self._regressor_knot_pooling_scale = list()
            self._regressor_knot_scale = list()
            return

        def _validate_params_len(params, valid_length):
            for p in params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

        # regressor defaults
        num_of_regressors = len(self.regressor_col)

        _validate_params_len([
            self.regressor_sign, self.regressor_knot_pooling_loc,
            self.regressor_knot_pooling_scale, self.regressor_knot_scale],
            num_of_regressors
        )

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * num_of_regressors

        if self.regressor_knot_pooling_loc is None:
            self._regressor_knot_pooling_loc = [DEFAULT_COEFFICIENTS_KNOT_POOL_LOC] * num_of_regressors

        if self.regressor_knot_pooling_scale is None:
            self._regressor_knot_pooling_scale = [DEFAULT_COEFFICIENTS_KNOT_POOL_SCALE] * num_of_regressors

        if self.regressor_knot_scale is None:
            self._regressor_knot_scale = [DEFAULT_COEFFICIENTS_KNOT_SCALE] * num_of_regressors

        self._num_of_regressors = num_of_regressors

    def _set_static_regression_attributes(self):
        # if no regressors, end here
        if self._num_of_regressors == 0:
            return

        for index, reg_sign in enumerate(self._regressor_sign):
            if reg_sign == '+':
                self._num_of_positive_regressors += 1
                self._positive_regressor_col.append(self.regressor_col[index])
                # used for 'pr_knot_loc' sampling in pyro
                self._positive_regressor_knot_pooling_loc.append(self._regressor_knot_pooling_loc[index])
                self._positive_regressor_knot_pooling_scale.append(self._regressor_knot_pooling_scale[index])
                # used for 'pr_knot' sampling in pyro
                self._positive_regressor_knot_scale.append(self._regressor_knot_scale[index])
            else:
                self._num_of_regular_regressors += 1
                self._regular_regressor_col.append(self.regressor_col[index])
                # used for 'rr_knot_loc' sampling in pyro
                self._regular_regressor_knot_pooling_loc.append(self._regressor_knot_pooling_loc[index])
                self._regular_regressor_knot_pooling_scale.append(self._regressor_knot_pooling_scale[index])
                # used for 'rr_knot' sampling in pyro
                self._regular_regressor_knot_scale.append(self._regressor_knot_scale[index])
        # regular first, then positive
        self._regressor_col = self._regular_regressor_col + self._positive_regressor_col

    @staticmethod
    def _validate_coef_prior(coef_prior_list):
        for test_dict in coef_prior_list:
            if set(test_dict.keys()) != set([
                CoefPriorDictKeys.NAME.value,
                CoefPriorDictKeys.PRIOR_START_TP_IDX.value,
                CoefPriorDictKeys.PRIOR_END_TP_IDX.value,
                CoefPriorDictKeys.PRIOR_MEAN.value,
                CoefPriorDictKeys.PRIOR_SD.value,
                CoefPriorDictKeys.PRIOR_REGRESSOR_COL.value
            ]):
                raise IllegalArgument('wrong key name in inserted prior dict')
            len_insert_prior = list()
            for key, val in test_dict.items():
                if key in [
                    CoefPriorDictKeys.PRIOR_MEAN.value,
                    CoefPriorDictKeys.PRIOR_SD.value,
                    CoefPriorDictKeys.PRIOR_REGRESSOR_COL.value,
                ]:
                    len_insert_prior.append(len(val))
            if not all(len_insert == len_insert_prior[0] for len_insert in len_insert_prior):
                raise IllegalArgument('wrong dimension length in inserted prior dict')

    @staticmethod
    def _validate_level_knot_inputs(level_knot_dates, level_knots):
        if len(level_knots) != len(level_knot_dates):
            raise IllegalArgument('level_knots and level_knot_dates should have the same length')

    @staticmethod
    def _get_gap_between_dates(start_date, end_date, freq):
        diff = end_date - start_date
        gap = np.array(diff / np.timedelta64(1, freq))

        return gap

    @staticmethod
    def _set_knots_tp(knots_distance, cutoff):
        # start in the middle
        knots_idx_start = round(knots_distance / 2)
        knots_idx = np.arange(knots_idx_start, cutoff, knots_distance)

        return knots_idx

    def _set_coef_prior_idx(self):
        if self._coef_prior_list and len(self._regressor_col) > 0:
            for x in self._coef_prior_list:
                prior_regressor_col_idx = [
                    np.where(np.array(self._regressor_col) == col)[0][0]
                    for col in x['prior_regressor_col']
                ]
                x.update({'prior_regressor_col_idx': prior_regressor_col_idx})

    def _set_static_attributes(self):
        """model data input based on args at instantiation or computed from args at instantiation"""
        self._set_default_args()
        self._set_static_regression_attributes()

        self._validate_level_knot_inputs(self.level_knot_dates, self.level_knots)

        if self._coef_prior_list:
            self._validate_coef_prior(self._coef_prior_list)
            self._set_coef_prior_idx()

    def _set_valid_response_attributes(self):
        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            if self.num_of_observations < max_seasonality:
                raise ModelException(
                    "Number of observations {} is less than max seasonality {}".format(
                        self.num_of_observations, max_seasonality))
        # get some reasonable offset to regularize response to make default priors scale-insensitive
        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            self.response_offset = np.nanmean(self.response[:max_seasonality])
        else:
            self.response_offset = np.nanmean(self.response)

        self.is_valid_response = ~np.isnan(self.response)
        # [0] to convert tuple back to array
        self.which_valid_response = np.where(self.is_valid_response)[0]
        self.num_of_valid_response = len(self.which_valid_response)

    def _set_regressor_matrix(self, df):
        # validate regression columns
        if self.regressor_col is not None and \
                not set(self.regressor_col).issubset(df.columns):
            raise ModelException(
                "DataFrame does not contain specified regressor column(s)."
            )

        # init of regression matrix depends on length of response vector
        self._positive_regressor_matrix = np.zeros((self.num_of_observations, 0), dtype=np.double)
        self._regular_regressor_matrix = np.zeros((self.num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self._num_of_positive_regressors > 0:
            self._positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col,).values

        if self._num_of_regular_regressors > 0:
            self._regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col,).values

    def _set_coefficients_kernel_matrix(self, df):
        # Note that our tp starts by 1; to convert back to index of array, reduce it by 1
        tp = np.arange(1, self.num_of_observations + 1) / self.num_of_observations
        # this approach put knots in full range
        self._cutoff = self.num_of_observations
        self._kernel_coefficients = np.zeros((self.num_of_observations, 0), dtype=np.double)
        self._num_knots_coefficients = 0

        # kernel of coefficients calculations
        # if self._knots_tp_coefficients is None:
        if self._num_of_regressors > 0:
            if self._coefficients_knot_dates is None:
                if self.coefficients_knot_length is not None:
                    # TODO: approximation; can consider directly coefficients_knot_length it as step size
                    knots_distance = self.coefficients_knot_length
                else:
                    number_of_knots = round(1 / self.span_coefficients)
                    knots_distance = math.ceil(self._cutoff / number_of_knots)

                knots_idx_coef = self._set_knots_tp(knots_distance, self._cutoff)
                self._knots_tp_coefficients = (1 + knots_idx_coef) / self.num_of_observations
                self._coefficients_knot_dates = df[self.date_col].values[knots_idx_coef]
            else:
                # FIXME: this only works up to daily series (not working on hourly series)
                self._coefficients_knot_dates = pd.to_datetime([
                    x for x in self._coefficients_knot_dates if (x <= df[self.date_col].max()) \
                                                                and (x >= df[self.date_col].min())
                ])
                if self.date_freq is None:
                    self.date_freq = pd.infer_freq(df[self.date_col])[0]
                start_date = self.training_start
                self._knots_tp_coefficients = np.array(
                    (self._get_gap_between_dates(start_date, self._coefficients_knot_dates, self.date_freq) + 1) /
                    (self._get_gap_between_dates(start_date, self.training_end, self.date_freq) + 1)
                )

            kernel_coefficients = gauss_kernel(tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            self._num_knots_coefficients = len(self._knots_tp_coefficients)
            self._kernel_coefficients = kernel_coefficients

    def _generate_tp(self, prediction_date_array):
        prediction_start = prediction_date_array[0]
        output_len = len(prediction_date_array)
        if prediction_start > self.training_end:
            start = self.num_of_observations
        else:
            start = pd.Index(self.date_array).get_loc(prediction_start)

        new_tp = np.arange(start + 1, start + output_len + 1) / self.num_of_observations
        return new_tp

    def _generate_insample_tp(self, date_array):
        idx = np.nonzero(np.in1d(self.date_array, date_array))[0]
        tp = (idx + 1) / self.num_of_observations
        return tp

    def _generate_coefs(self, prediction_date_array, coef_knot_dates, coef_knot):
        new_tp = self._generate_tp(prediction_date_array)
        knots_tp_coef = self._generate_insample_tp(coef_knot_dates)
        kernel_coef = sandwich_kernel(new_tp, knots_tp_coef)
        # kernel_coef = gauss_kernel(new_tp, knots_tp_coef, rho)
        # kernel_coef = kernel_coef / np.sum(kernel_coef, axis=1, keepdims=True)
        coefs = np.squeeze(np.matmul(coef_knot, kernel_coef.transpose(1, 0)), axis=0).transpose(1, 0)
        return coefs

    def _generate_seas(self, df, coef_knot_dates, coef_knot, seasonality, seasonality_fs_order):
        """To calculate the seasonality term based on the _seasonal_knots_input.
        :param df: input df
        :param coef_knot_dates: dates for coef knots
        :param coef_knot: knot values for coef
        :param seasonality: seasonality input
        :param seasonality_fs_order: seasonality_fs_order input
        :return:
        """
        prediction_date_array = df[self.date_col].values
        prediction_start = prediction_date_array[0]

        df = df.copy()
        if prediction_start > self.training_end:
            forecast_dates = set(prediction_date_array)
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = self.num_of_observations
        else:
            # compute how many steps to forecast
            forecast_dates = set(prediction_date_array) - set(self.date_array)
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or \
                               - (len(set(self.date_array) - set(prediction_date_array)))
            # time index for prediction start
            start = pd.Index(self.date_array).get_loc(prediction_start)

        fs_cols = []
        for idx, s in enumerate(seasonality):
            order = seasonality_fs_order[idx]
            df, fs_cols_temp = make_fourier_series_df(df, s, order=order, prefix='seas{}_'.format(s), shift=start)
            fs_cols += fs_cols_temp

        sea_regressor_matrix = df.filter(items=fs_cols).values
        sea_coefs = self._generate_coefs(prediction_date_array, coef_knot_dates, coef_knot)
        seas = np.sum(sea_coefs * sea_regressor_matrix, axis=-1)

        return seas

    def _set_levs_and_seas(self, df):
        tp = np.arange(1, self.num_of_observations + 1) / self.num_of_observations
        # trim level knots dates when they are beyond training dates
        lev_knot_dates = list()
        lev_knots = list()
        # TODO: any faster way instead of a simple loop?
        for i, x in enumerate(self.level_knot_dates):
            if (x <= df[self.date_col].max()) and (x >= df[self.date_col].min()):
                lev_knot_dates.append(x)
                lev_knots.append(self.level_knots[i])
        self._level_knot_dates = pd.to_datetime(lev_knot_dates)
        self._level_knots = np.array(lev_knots)
        infer_freq = pd.infer_freq(df[self.date_col])[0]
        start_date = self.training_start

        if len(self.level_knots) > 0 and len(self.level_knot_dates) > 0:
            self.knots_tp_level = np.array(
                (self._get_gap_between_dates(start_date, self._level_knot_dates, infer_freq) + 1) /
                (self._get_gap_between_dates(start_date, self.training_end, infer_freq) + 1)
            )
        else:
            raise ModelException("User need to supply a list of level knots.")

        kernel_level = sandwich_kernel(tp, self.knots_tp_level)
        self._kernel_level = kernel_level
        self._num_knots_level = len(self._level_knot_dates)

        if self._seasonal_knots_input is not None:
            self._seas_term = self._generate_seas(
                df,
                self._seasonal_knots_input['_seas_coef_knot_dates'],
                self._seasonal_knots_input['_sea_coef_knot'],
                # self._seasonal_knots_input['_sea_rho'],
                self._seasonal_knots_input['_seasonality'],
                self._seasonal_knots_input['_seasonality_fs_order'])

    def _filter_coef_prior(self, df):
        if self._coef_prior_list and len(self._regressor_col) > 0:
            # iterate over a copy due to the removal operation
            for test_dict in self._coef_prior_list[:]:
                prior_regressor_col = test_dict['prior_regressor_col']
                m = test_dict['prior_mean']
                sd = test_dict['prior_sd']
                end_tp_idx = min(test_dict['prior_end_tp_idx'], df.shape[0])
                start_tp_idx = min(test_dict['prior_start_tp_idx'], df.shape[0])
                if start_tp_idx < end_tp_idx:
                    expected_shape = (end_tp_idx - start_tp_idx, len(prior_regressor_col))
                    test_dict.update({'prior_end_tp_idx': end_tp_idx})
                    test_dict.update({'prior_start_tp_idx': start_tp_idx})
                    # mean/sd expanding
                    test_dict.update({'prior_mean': np.full(expected_shape, m)})
                    test_dict.update({'prior_sd': np.full(expected_shape, sd)})
                else:
                    # removing invalid prior
                    self._coef_prior_list.remove(test_dict)

    def _set_dynamic_attributes(self, df):
        """Overriding: func: `~orbit.models.BaseETS._set_dynamic_attributes"""
        self._set_valid_response_attributes()
        self._set_regressor_matrix(df)
        self._set_coefficients_kernel_matrix(df)
        self._set_levs_and_seas(df)
        self._filter_coef_prior(df)

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

    def _predict(self, posterior_estimates, df, include_error=False, decompose=False, **kwargs):
        """Vectorized version of prediction math"""
        ################################################################
        # Model Attributes
        ################################################################

        model = deepcopy(posterior_estimates)
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        ################################################################
        # Prediction Attributes
        ################################################################
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

        if prediction_df_meta['prediction_start'] < self.training_start:
            raise PredictionException('Prediction start must be after training start.')

        output_len = prediction_df_meta['df_length']
        date_array = prediction_df_meta['date_array']
        prediction_start = prediction_df_meta['prediction_start']

        # Here assume dates are ordered and consecutive
        # if prediction_df_meta['prediction_start'] > self.training_end,
        # assume prediction starts right after train end
        if prediction_start > self.training_end:
            # time index for prediction start
            start = self.num_of_observations
        else:
            start = pd.Index(self.date_array).get_loc(prediction_start)

        new_tp = np.arange(start + 1, start + output_len + 1) / self.num_of_observations
        if include_error:
            # in-sample knots
            lev_knot_in = model.get(constants.BaseSamplingParameters.LEVEL_KNOT.value)
            # TODO: hacky way; let's just assume last two knot distance is knots distance for all knots
            lev_knot_width = self.knots_tp_level[-1] - self.knots_tp_level[-2]
            # check whether we need to put new knots for simulation
            if new_tp[-1] >= self.knots_tp_level[-1] + lev_knot_width:
                # derive knots tp
                knots_tp_level_out = np.arange(self.knots_tp_level[-1] + lev_knot_width, new_tp[-1], lev_knot_width)
                new_knots_tp_level = np.concatenate([self.knots_tp_level, knots_tp_level_out])
                lev_knot_out = np.random.laplace(0, self.level_knot_scale,
                                                 size=(lev_knot_in.shape[0], len(knots_tp_level_out)))
                lev_knot_out = np.cumsum(np.concatenate([lev_knot_in[:, -1].reshape(-1, 1), lev_knot_out],
                                                        axis=1), axis=1)[:, 1:]
                lev_knot = np.concatenate([lev_knot_in, lev_knot_out], axis=1)
            else:
                new_knots_tp_level = self.knots_tp_level
                lev_knot = lev_knot_in
            kernel_level = sandwich_kernel(new_tp, new_knots_tp_level)
        else:
            lev_knot = model.get(constants.BaseSamplingParameters.LEVEL_KNOT.value)
            kernel_level = sandwich_kernel(new_tp, self.knots_tp_level)
        obs_scale = model.get(constants.BaseSamplingParameters.OBS_SCALE.value)
        obs_scale = obs_scale.reshape(-1, 1)

        if self._seasonal_knots_input is not None:
            seas = self._generate_seas(df,
                                       self._seasonal_knots_input['_seas_coef_knot_dates'],
                                       self._seasonal_knots_input['_sea_coef_knot'],
                                       # self._seasonal_knots_input['_sea_rho'],
                                       self._seasonal_knots_input['_seasonality'],
                                       self._seasonal_knots_input['_seasonality_fs_order'])
            # seas is 1-d array, add the batch size back
            seas = np.expand_dims(seas, 0)
        else:
            # follow component shapes
            seas = np.zeros((1, output_len))

        trend = np.matmul(lev_knot, kernel_level.transpose(1, 0))
        regression = np.zeros(trend.shape)
        if self._num_of_regressors > 0:
            coef_knot = model.get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)

            # init of regression matrix depends on length of response vector
            pred_regular_regressor_matrix = np.zeros((output_len, 0), dtype=np.double)
            pred_positive_regressor_matrix = np.zeros((output_len, 0), dtype=np.double)

            # update regression matrices
            if self._num_of_regular_regressors > 0:
                pred_regular_regressor_matrix = df.filter(
                    items=self._regular_regressor_col,).values
            if self._num_of_positive_regressors > 0:
                pred_positive_regressor_matrix = df.filter(
                    items=self._positive_regressor_col,).values
            # regular first, then positive
            pred_regressor_matrix = np.concatenate([pred_regular_regressor_matrix,
                                                    pred_positive_regressor_matrix], axis=-1)
            kernel_coefficients = gauss_kernel(new_tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            regression = np.sum(
                np.matmul(coef_knot, kernel_coefficients.transpose(1, 0)) *
                pred_regressor_matrix.transpose(1, 0), axis=-2
            )

        if include_error:
            epsilon = nct.rvs(self.degree_of_freedom, nc=0, loc=0,
                              scale=obs_scale, size=(num_sample, len(new_tp)))
            pred_array = trend + seas + regression + epsilon
        else:
            pred_array = trend + seas + regression

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend,
                'seasonality_input': seas,
                'regression': regression
            }
        else:
            decomp_dict = {'prediction': pred_array}

        return decomp_dict

    def get_regression_coefs(self, aggregate_method, include_ci=False, date_array=None,
                             lower=0.05, upper=0.95):
        """Return DataFrame regression coefficients

        If PredictMethod is `full` return `mean` of coefficients instead
        """
        # init dataframe
        reg_df = pd.DataFrame()
        # end if no regressors
        if self._num_of_regular_regressors + self._num_of_positive_regressors == 0:
            return reg_df

        if date_array is None:
            # if date_array not specified, dynamic coefficients in the training perior will be retrieved
            reg_df[self.date_col] = self.date_array
            coef_knots = self._aggregate_posteriors \
                .get(aggregate_method) \
                .get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
            regressor_betas = np.squeeze(np.matmul(coef_knots, self._kernel_coefficients.transpose(1, 0)), axis=0)
            regressor_betas = regressor_betas.transpose(1, 0)
        else:
            date_array = pd.to_datetime(date_array).values
            if not is_ordered_datetime(date_array):
                raise IllegalArgument('Datetime index must be ordered and not repeat')
            reg_df[self.date_col] = date_array

            # TODO: validate that all regressor columns are present, if any
            prediction_start = date_array[0]
            if prediction_start < self.training_start:
                raise PredictionException('Prediction start must be after training start.')

            output_len = len(date_array)

            # If we cannot find a match of prediction range, assume prediction starts right after train end
            if prediction_start > self.training_end:
                # time index for prediction start
                start = self.num_of_observations
            else:
                # time index for prediction start
                start = pd.Index(self.date_array).get_loc(prediction_start)

            new_tp = np.arange(start + 1, start + output_len + 1) / self.num_of_observations

            kernel_coefficients = gauss_kernel(new_tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            coef_knots = self._aggregate_posteriors \
                .get(aggregate_method) \
                .get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
            regressor_betas = np.squeeze(np.matmul(coef_knots, kernel_coefficients.transpose(1, 0)), axis=0)
            regressor_betas = regressor_betas.transpose(1, 0)

        # get column names
        rr_cols = self._regular_regressor_col
        pr_cols = self._positive_regressor_col
        regressor_col = rr_cols + pr_cols
        for idx, col in enumerate(regressor_col):
            reg_df[col] = regressor_betas[:, idx]

        if include_ci:
            posterior_samples = self._posterior_samples
            param_ndarray = posterior_samples.get(constants.RegressionSamplingParameters.COEFFICIENTS.value)
            coefficients_lower = np.quantile(param_ndarray, [lower], axis=0)
            coefficients_upper = np.quantile(param_ndarray, [upper], axis=0)
            coefficients_lower = np.squeeze(coefficients_lower, axis=0)
            coefficients_upper = np.squeeze(coefficients_upper, axis=0)

            reg_df_lower = reg_df.copy()
            reg_df_upper = reg_df.copy()
            for idx, col in enumerate(regressor_col):
                reg_df_lower[col] = coefficients_lower[:, idx]
                reg_df_upper[col] = coefficients_upper[:, idx]
            return reg_df, reg_df_lower, reg_df_upper

        return reg_df

    def get_regression_coef_knots(self, aggregate_method):
        """Return DataFrame regression coefficient knots

        If PredictMethod is `full` return `mean` of coefficient knots instead
        """
        # init dataframe
        knots_df = pd.DataFrame()
        # end if no regressors
        if self._num_of_regular_regressors + self._num_of_positive_regressors == 0:
            return knots_df

        knots_df[self.date_col] = self._coefficients_knot_dates
        coef_knots = self._aggregate_posteriors \
            .get(aggregate_method) \
            .get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)

        # get column names
        rr_cols = self._regular_regressor_col
        pr_cols = self._positive_regressor_col
        regressor_col = rr_cols + pr_cols
        for idx, col in enumerate(regressor_col):
            knots_df[col] = np.transpose(coef_knots[:, idx])

        return knots_df

    def plot_regression_coefs(self,
                              coef_df,
                              coef_df_lower=None,
                              coef_df_upper=None,
                              ncol=2,
                              figsize=None,
                              ylim=None):
        """Plot regression coefficients
        """
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

        return axes


class KTRXFull(FullBayesianTemplate, BaseKTRX):
    """Concrete KTRX model for full Bayesian prediction"""
    _supported_estimator_types = [PyroEstimatorVI]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_regression_coefs(self, aggregate_method='mean', include_ci=False, date_array=None, lower=0.05, upper=0.95):
        self._set_aggregate_posteriors()
        return super().get_regression_coefs(aggregate_method=aggregate_method,
                                            include_ci=include_ci,
                                            date_array=date_array,
                                            lower=lower,
                                            upper=upper)

    def get_regression_coef_knots(self, aggregate_method='mean'):
        self._set_aggregate_posteriors()
        return super().get_regression_coef_knots(aggregate_method=aggregate_method)

    def plot_regression_coefs(self, aggregate_method='mean', include_ci=False, date_array=None, **kwargs):
        if include_ci:
            coef_df, coef_df_lower, coef_df_upper = self.get_regression_coefs(
                aggregate_method=aggregate_method,
                include_ci=True
            )
        else:
            coef_df = self.get_regression_coefs(aggregate_method=aggregate_method,
                                                include_ci=False,
                                                date_array=date_array)
            coef_df_lower = None
            coef_df_upper = None

        return super().plot_regression_coefs(coef_df=coef_df,
                                             coef_df_lower=coef_df_lower,
                                             coef_df_upper=coef_df_upper,
                                             **kwargs)

class KTRXAggregated(AggregatedPosteriorTemplate, BaseKTRX):
    """Concrete KTRX model for aggregated Bayesian prediction"""
    _supported_estimator_types = [PyroEstimatorVI]

    def __init__(self, aggregate_method='mean', **kwargs):
        super().__init__(**kwargs)

    def get_regression_coefs(self, date_array=None):
        return super().get_regression_coefs(aggregate_method=self.aggregate_method,
                                            include_ci=False,
                                            date_array=date_array)

    def get_regression_coef_knots(self):
        return super().get_regression_coef_knots(aggregate_method=self.aggregate_method)

    def plot_regression_coefs(self, date_array=None, **kwargs):
        coef_df = self.get_regression_coefs(date_array=date_array)
        return super().plot_regression_coefs(coef_df=coef_df,
                                             **kwargs)
