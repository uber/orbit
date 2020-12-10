import pandas as pd
import numpy as np
import math
from scipy.stats import nct
import torch
from copy import copy, deepcopy
import matplotlib.pyplot as plt

from ..constants import ktrx as constants
from ..constants.constants import (
    PredictMethod
)
from ..constants.ktrx import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_COEFFICIENTS_KNOT_POOL_SCALE,
    DEFAULT_COEFFICIENTS_KNOT_POOL_LOC,
    DEFAULT_COEFFICIENTS_KNOT_SCALE,
)

from ..estimators.pyro_estimator import PyroEstimatorVI
from ..exceptions import IllegalArgument, ModelException, PredictionException
from .base_model import BaseModel
from ..utils.general import is_ordered_datetime
from ..utils.kernels import gauss_kernel, sandwich_kernel
from ..utils.features import make_fourier_series_df


def generate_tp(prediction_date_array, training_df_meta):
    # should be you prediction date array
    prediction_start = prediction_date_array[0]
    trained_len = training_df_meta['df_length']
    output_len = len(prediction_date_array)
    if prediction_start > training_df_meta['training_end']:
        start = trained_len
    else:
        start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

    new_tp = np.arange(start + 1, start + output_len + 1) / trained_len
    return new_tp


def generate_insample_tp(date_array, training_df_meta):
    idx = np.nonzero(np.in1d(training_df_meta['date_array'], date_array))[0]
    tp = (idx + 1) / training_df_meta['df_length']
    return tp


def generate_levs(prediction_date_array, training_df_meta, level_knot_dates, lev_knot):
    new_tp = generate_tp(prediction_date_array, training_df_meta)
    knots_tp_level = generate_insample_tp(level_knot_dates, training_df_meta)
    kernel_level = sandwich_kernel(new_tp, knots_tp_level)
    levs = np.matmul(lev_knot, kernel_level.transpose(1, 0))
    return levs


def generate_coefs(prediction_date_array, training_df_meta, coef_knot_dates, coef_knot, rho):
    new_tp = generate_tp(prediction_date_array, training_df_meta)
    knots_tp_coef = generate_insample_tp(coef_knot_dates, training_df_meta)
    kernel_coef = gauss_kernel(new_tp, knots_tp_coef, rho)
    kernel_coef = kernel_coef / np.sum(kernel_coef, axis=1, keepdims=True)
    coefs = np.squeeze(np.matmul(coef_knot, kernel_coef.transpose(1, 0)), axis=0).transpose(1, 0)
    return coefs


def generate_seas(df, date_col, training_df_meta, coef_knot_dates, coef_knot, rho, seasonality, seasonality_fs_order):
    prediction_date_array = df[date_col].values
    prediction_start = prediction_date_array[0]
    trained_len = training_df_meta['df_length']
    df = df.copy()
    if prediction_start > training_df_meta['training_end']:
        forecast_dates = set(prediction_date_array)
        n_forecast_steps = len(forecast_dates)
        # time index for prediction start
        start = trained_len
    else:
        # compute how many steps to forecast
        forecast_dates = set(prediction_date_array) - set(training_df_meta['date_array'])
        # check if prediction df is a subset of training df
        # e.g. "negative" forecast steps
        n_forecast_steps = len(forecast_dates) or \
                           - (len(set(training_df_meta['date_array']) - set(prediction_date_array)))
        # time index for prediction start
        start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

    fs_cols = []
    for idx, s in enumerate(seasonality):
        order = seasonality_fs_order[idx]
        df, fs_cols_temp = make_fourier_series_df(df, s, order=order, prefix='seas{}_'.format(s), shift=start)
        fs_cols += fs_cols_temp

    sea_regressor_matrix = df.filter(items=fs_cols).values
    sea_coefs = generate_coefs(prediction_date_array, training_df_meta, coef_knot_dates, coef_knot, rho)
    seas = np.sum(sea_coefs * sea_regressor_matrix, axis=-1)

    return seas

class BaseKTRX(BaseModel):
    """Base LGT model object with shared functionality for Full, Aggregated, and MAP methods

    Parameters
    ----------
    response_col : str
        response column name
    date_col : str
        date column name
    seasonality : int, or list of int
        multiple seasonality
    seasonality_fs_order : int, or list of int
        fourier series order for seasonality
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
    span_level : float between (0, 1)
        window width to decide the number of windows for the level (trend) term.
        e.g., span 0.1 will produce 10 windows.
    span_coefficients : float between (0, 1)
        window width to decide the number of windows for the regression term
    rho_level : float
        sigma in the Gaussian kernel for the level term
    rho_coefficients : float
        sigma in the Gaussian kernel for the regression term
    insert_prior_regressor_col : list
        list of regressor names to ingest priors
    insert_prior_tp_idx : list
        list of time points to ingest priors
    insert_prior_mean : list
        list of ingested prior means
    insert_prior_sd : list
        list of ingested prior sigmas

    """
    _data_input_mapper = constants.DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'ktrx'
    _supported_estimator_types = None  # set for each model

    def __init__(self,
                 response_col='y',
                 date_col='ds',
                 level_knot_scale=0.1,
                 regressor_col=None,
                 regressor_sign=None,
                 regressor_knot_pooling_loc=None,
                 regressor_knot_pooling_scale=None,
                 regressor_knot_scale=None,
                 span_level=0.1,
                 rho_level=0.05,
                 span_coefficients=0.2,
                 rho_coefficients=0.15,
                 degree_of_freedom=30,
                 # coef priors on specific time-point
                 insert_prior_regressor_col=None,
                 insert_prior_tp_idx=None,
                 insert_prior_mean=None,
                 insert_prior_sd=None,
                 # knot customization
                 level_knot_dates=None,
                 level_knots=None,
                 seasonal_knots_input=None,
                 **kwargs):
        super().__init__(**kwargs)  # create estimator in base class
        self.response_col = response_col
        self.date_col = date_col

        self.level_knot_scale = level_knot_scale
        self.regressor_col = regressor_col
        self.regressor_sign = regressor_sign
        self.regressor_knot_pooling_loc = regressor_knot_pooling_loc
        self.regressor_knot_pooling_scale = regressor_knot_pooling_scale
        self.regressor_knot_scale = regressor_knot_scale

        self.degree_of_freedom = degree_of_freedom

        self.span_level = span_level
        self.span_coefficients = span_coefficients
        self.rho_level = rho_level
        self.rho_coefficients = rho_coefficients

        self.insert_prior_regressor_col = insert_prior_regressor_col
        self.insert_prior_tp_idx = insert_prior_tp_idx
        self.insert_prior_mean = insert_prior_mean
        self.insert_prior_sd = insert_prior_sd

        self.level_knot_dates = level_knot_dates
        self.level_knots = level_knots
        self._seasonal_knots_input = seasonal_knots_input

        # set private var to arg value
        # if None set default in _set_default_base_args()
        self._regressor_sign = self.regressor_sign
        self._regressor_knot_pooling_loc = self.regressor_knot_pooling_loc
        self._regressor_knot_pooling_scale= self.regressor_knot_pooling_scale
        self._regressor_knot_scale = self.regressor_knot_scale

        self._insert_prior_regressor_col = self.insert_prior_regressor_col
        # self._insert_prior_idx = self.insert_prior_idx
        self._insert_prior_tp_idx = self.insert_prior_tp_idx
        self._insert_prior_mean = self.insert_prior_mean
        self._insert_prior_sd = self.insert_prior_sd
        self._insert_prior_idx = list()
        self._num_insert_prior = None

        self._level_knot_dates = self.level_knot_dates
        self._level_knots = self.level_knots
        self._num_knots_level = None
        self._knots_tp_level = None
        self._coef_knot_dates = None
        self._seas_term = None

        self._model_param_names = list()
        self._training_df_meta = None
        self._model_data_input = dict()

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
        self._response_mean = 0.0
        self._is_valid_response = None
        self._which_valid_response = None
        self._num_of_valid_response = 0
        self._seasonality = None

        # regression data
        self._knots_tp_coefficients = None
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

    def _validate_supported_estimator_type(self):
        if self.estimator_type not in self._supported_estimator_types:
            msg_template = "Model class: {} is incompatible with Estimator: {}"
            model_class = type(self)
            estimator_type = self.estimator_type
            raise IllegalArgument(msg_template.format(model_class, estimator_type))

    def _set_default_base_args(self):
        """Set default attributes for None
        """
        if self.insert_prior_regressor_col is None:
            self._insert_prior_regressor_col = list()
        if self.insert_prior_tp_idx is None:
            self._insert_prior_tp_idx = list()
        if self.insert_prior_mean is None:
            self._insert_prior_mean = list()
        if self.insert_prior_sd is None:
            self._insert_prior_sd = list()
        if self._num_insert_prior is None:
            self._num_insert_prior = len(self._insert_prior_tp_idx)
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

        def _validate(regression_params, valid_length):
            for p in regression_params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

        def _validate_insert_prior(insert_prior_params):
            len_insert_prior = list()
            for p in insert_prior_params:
                len_insert_prior.append(len(p))
            if not all(len_insert == len_insert_prior[0] for len_insert in len_insert_prior):
                raise IllegalArgument('Wrong dimension length in Insert Prior Input')

        def _validate_level_knot_inputs(level_knot_dates, level_knots):
            if len(level_knots) != len(level_knot_dates):
                raise IllegalArgument('level_knots and level_knot_dates should have the same length')

        # regressor defaults
        num_of_regressors = len(self.regressor_col)

        _validate(
            [self.regressor_sign, self.regressor_knot_pooling_loc,
             self.regressor_knot_pooling_scale, self.regressor_knot_scale],
            num_of_regressors
        )
        _validate_insert_prior([self._insert_prior_regressor_col, self._insert_prior_tp_idx,
                                self._insert_prior_mean, self._insert_prior_sd])

        _validate_level_knot_inputs(self.level_knot_dates, self.level_knots)

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

    def _set_insert_prior_idx(self):
        if self._num_insert_prior > 0 and len(self._regressor_col) > 0:
            for col in self._insert_prior_regressor_col:
                self._insert_prior_idx.append(np.where(np.array(self._regressor_col) == col)[0][0])

    def _set_static_data_attributes(self):
        """model data input based on args at instantiation or computed from args at instantiation"""
        self._set_default_base_args()
        self._set_static_regression_attributes()
        self._set_insert_prior_idx()

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

    def _set_training_df_meta(self, df):
        # Date Metadata
        # TODO: use from constants for dict key
        self._training_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'training_start': df[self.date_col].iloc[0],
            'training_end': df[self.date_col].iloc[-1]
        }

    def _set_valid_response_attributes(self):
        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            self._response_mean = np.nanmean(self._response[:max_seasonality])
        else:
            self._response_mean = np.nanmean(self._response)

        self._is_valid_response = ~np.isnan(self._response)
        # [0] to convert tuple back to array
        self._which_valid_response = np.where(self._is_valid_response)[0]
        self._num_of_valid_response = len(self._which_valid_response)
        self._response_sd = np.nanstd(self._response)

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

    def _set_coefficients_kernel_matrix(self, df):
        # Note that our tp starts by 1; to convert back to index of array, reduce it by 1
        tp = np.arange(1, self._num_of_observations + 1) / self._num_of_observations
        # this approach put knots in full range
        self._cutoff = self._num_of_observations
        # cutoff last 20%
        # self._cutoff = round(0.2 * self._num_of_observations)
        self._kernel_coefficients = np.zeros((self._num_of_observations, 0), dtype=np.double)
        if self._num_of_regressors > 0:
            # kernel of coefficients calculations
            # if self._knots_tp_coefficients is None:
            number_of_knots = round(1 / self.span_coefficients)
            knots_distance = math.ceil(self._cutoff / number_of_knots)
            # start in the middle
            knots_idx_start_coef = round(knots_distance / 2)
            knots_idx_coef = np.arange(knots_idx_start_coef, self._cutoff,  knots_distance)
            self._knots_tp_coefficients = (1 + knots_idx_coef) / self._num_of_observations
            self._coef_knot_dates = df[self.date_col].values[knots_idx_coef]

            kernel_coefficients = gauss_kernel(tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            self._num_knots_coefficients = len(self._knots_tp_coefficients)
            self._kernel_coefficients = kernel_coefficients

    def _set_levs_and_seas(self, df):
        tp = np.arange(1, self._num_of_observations + 1) / self._num_of_observations
        # trim level knots dates when they are beyond training dates
        self._level_knot_dates = pd.to_datetime([x for x in self.level_knot_dates if x <= df[self.date_col].max()])
        self._level_knots = self.level_knots[:len(self._level_knot_dates)]

        if len(self.level_knots) > 0 and len(self.level_knot_dates) > 0:
            # new_tp = generate_tp(self.level_knot_dates, self._training_df_meta)
            # self._knots_tp_level = generate_insample_tp(self._training_df_meta['date_array'], self._training_df_meta)
            self._knots_tp_level = np.array(
                ((self._level_knot_dates - self._training_df_meta['training_start']).days + 1) /
                ((self._training_df_meta['training_end'] - self._training_df_meta['training_start']).days + 1)
            )
        else:
            raise ModelException("User need to supply list of level knots.")

        kernel_level = sandwich_kernel(tp, self._knots_tp_level)

        self._kernel_level = kernel_level
        self._num_knots_level = len(self._level_knot_dates)

        if self._seasonal_knots_input is not None:
            self._seas_term = generate_seas(
                df, self.date_col, self._training_df_meta,
                self._seasonal_knots_input['_seas_coef_knot_dates'],
                self._seasonal_knots_input['_sea_coef_knot'],
                self._seasonal_knots_input['_sea_rho'],
                self._seasonal_knots_input['_seasonality'],
                self._seasonal_knots_input['_seasonality_fs_order'])

    def _set_dynamic_data_attributes(self, df):
        """data input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)

        self._validate_training_df(df)
        self._set_training_df_meta(df)
        self._set_valid_response_attributes()
        self._set_regressor_matrix(df)
        self._set_coefficients_kernel_matrix(df)
        self._set_levs_and_seas(df)

    def _set_model_param_names(self):
        """Model parameters to extract"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]
        self._model_param_names += [param.value for param in constants.RegressionSamplingParameters]

    def _get_model_param_names(self):
        return self._model_param_names

    def _set_model_data_input(self):
        """Collects data attributes into a dict for sampling"""
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

    def _predict(self, posterior_estimates, df, include_error=False, decompose=False, random_state=None):
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
        date_array = prediction_df_meta['date_array']
        prediction_start = prediction_df_meta['prediction_start']

        # Here assume dates are ordered and consecutive
        # if prediction_df_meta['prediction_start'] > training_df_meta['training_end'],
        # assume prediction starts right after train end

        # If we cannot find a match of prediction range, assume prediction starts right after train
        # end
        if prediction_start > training_df_meta['training_end']:
            forecast_dates = set(date_array)
            start = trained_len
        else:
            # compute how many steps to forecast
            forecast_dates = set(date_array) - set(training_df_meta['date_array'])
            # time index for prediction start
            start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

        new_tp = np.arange(start + 1, start + output_len + 1) / trained_len

        # Replacing this ----
        # TODO: check utility?
        # gap_time = prediction_df_meta['prediction_start'] - training_df_meta['training_start']
        # infer_freq = pd.infer_freq(df[self.date_col])[0]
        # gap_int = int(gap_time / np.timedelta64(1, infer_freq))
        #
        # # 1. set idx = test_df[date_col]
        # # 2. search match position of idx[0], set position = pos
        # # 3. if match, start from pos, assume all time points ordered so that your prediction horizon = pos + len(test_df)
        # # 3b. if not match, assumne start from last idx + 1, perdiction horizon = len(train_df) + len(test_df)
        #
        # new_tp = np.arange(1 + gap_int, output_len + gap_int + 1)
        # new_tp = new_tp / trained_len
        # Replacing this ---- END

        kernel_level = sandwich_kernel(new_tp, self._knots_tp_level)
        kernel_coefficients = gauss_kernel(new_tp, self._knots_tp_coefficients, rho=self.rho_coefficients)

        lev_knot = model.get(constants.BaseSamplingParameters.LEVEL_KNOT.value)
        coef_knot = model.get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
        obs_scale = model.get(constants.BaseSamplingParameters.OBS_SCALE.value)
        obs_scale = obs_scale.reshape(-1, 1)

        if self._seasonal_knots_input is not None:
            seas = generate_seas(df, self.date_col, self._training_df_meta,
                                        self._seasonal_knots_input['_seas_coef_knot_dates'],
                                        self._seasonal_knots_input['_sea_coef_knot'],
                                        self._seasonal_knots_input['_sea_rho'],
                                        self._seasonal_knots_input['_seasonality'],
                                        self._seasonal_knots_input['_seasonality_fs_order'])
        else:
            seas = 0.0

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

        trend = np.matmul(lev_knot, kernel_level.transpose(1, 0))
        regression = np.zeros(trend.shape)
        if self._num_of_regressors > 0:
            regression = np.sum(np.matmul(coef_knot, kernel_coefficients.transpose(1, 0)) * \
                                pred_regressor_matrix.transpose(1, 0), axis=-2)
        if include_error:
            epsilon = nct.rvs(self.degree_of_freedom, nc=0, loc=0,
                              scale=obs_scale, size=(num_sample, len(new_tp)), random_state=random_state)
            pred_array = trend + seas + regression + epsilon
        else:
            pred_array = trend + seas + regression

        # if decompose output dictionary of components
        if decompose:
            if self._seasonal_knots_input is not None:
                decomp_dict = {
                    'prediction': pred_array,
                    'trend': trend,
                    'seasonality_input': seas,
                    'regression': regression
                }
            else:
                decomp_dict = {
                    'prediction': pred_array,
                    'trend': trend,
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

    def get_regression_coefs(self, aggregate_method, include_ci=False, date_array=None):
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
            reg_df[self.date_col] = self._training_df_meta['date_array']
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
            training_df_meta = self._training_df_meta
            prediction_start = date_array[0]
            if prediction_start < training_df_meta['training_start']:
                raise PredictionException('Prediction start must be after training start.')

            trained_len = training_df_meta['df_length'] # i.e., self._num_of_observations
            output_len = len(date_array)

            # If we cannot find a match of prediction range, assume prediction starts right after train
            # end
            if prediction_start > training_df_meta['training_end']:
                forecast_dates = set(date_array)
                # time index for prediction start
                start = trained_len
            else:
                # compute how many steps to forecast
                forecast_dates = set(date_array) - set(training_df_meta['date_array'])
                # time index for prediction start
                start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

            new_tp = np.arange(start + 1, start + output_len + 1) / trained_len

            # Here assume dates are ordered and consecutive
            # if prediction_df_meta['prediction_start'] > training_df_meta['training_end'],
            # assume prediction starts right after train end
            # # TODO: check utility?
            # gap_time = prediction_start - training_df_meta['training_start']
            # infer_freq = pd.infer_freq(training_df_meta['date_array'])[0]
            # gap_int = int(gap_time / np.timedelta64(1, infer_freq))
            #
            # new_tp = np.arange(1 + gap_int, output_len + gap_int + 1)
            # new_tp = new_tp / trained_len
            kernel_coefficients = gauss_kernel(new_tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            coef_knots = self._aggregate_posteriors \
                .get(aggregate_method) \
                .get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
            # TODO: this looks sub-optimal; let's simplify this later
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

        return axes


class KTRXFull(BaseKTRX):
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

    def _bootstrap(self, n, random_state=None):
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
        if random_state is not None:
            np.random.seed(random_state)
        sample_idx = np.random.choice(
            range(num_samples),
            size=n,
            replace=True,
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

    def predict(self, df, decompose=False, random_state=None):
        """Return model predictions as a function of fitted model and df"""
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(self._n_bootstrap_draws, random_state=random_state) \
            if self._n_bootstrap_draws > 1 \
            else self._posterior_samples

        predicted_dict = self._predict(
            posterior_estimates=posterior_samples,
            df=df,
            include_error=True,
            random_state=random_state,
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

    def get_regression_coefs(self, aggregate_method='mean', include_ci=False, date_array=None):
        self._set_aggregate_posteriors()
        return super().get_regression_coefs(aggregate_method=aggregate_method,
                                            include_ci=include_ci,
                                            date_array=date_array)

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


class KTRXAggregated(BaseKTRX):
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

    def predict(self, df, decompose=False):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(self.aggregate_method)

        predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors,
            df=df,
            include_error=False,
            decompose=decompose,
        )

        # must flatten to convert to DataFrame
        for k, v in predicted_dict.items():
            predicted_dict[k] = v.flatten()

        predicted_df = pd.DataFrame(predicted_dict)
        predicted_df = self._prepend_date_column(predicted_df, df)

        return predicted_df

    def get_regression_coefs(self, date_array=None):
        return super().get_regression_coefs(aggregate_method=self.aggregate_method,
                                            include_ci=False,
                                            date_array=date_array)

    def plot_regression_coefs(self, date_array=None, **kwargs):
        coef_df = self.get_regression_coefs(date_array=date_array)
        return super().plot_regression_coefs(coef_df=coef_df,
                                             **kwargs)