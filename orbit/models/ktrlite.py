import pandas as pd
import numpy as np
import math
from scipy.stats import nct
import torch
from copy import copy, deepcopy

from ..constants import ktrlite as constants
from ..constants.constants import (
    PredictMethod
)

from ..estimators.stan_estimator import StanEstimatorMAP
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..utils.general import is_ordered_datetime
from ..utils.kernels import gauss_kernel, sandwich_kernel
from ..utils.features import make_fourier_series_df
from .template import BaseTemplate, MAPTemplate


class BaseKTRLite(BaseTemplate):
    """Base LGT model object with shared functionality for Full, Aggregated, and MAP methods

    Parameters
    ----------
    seasonality : int, or list of int
        multiple seasonality
    seasonality_fs_order : int, or list of int
        fourier series order for seasonality
    level_knot_scale : float
        sigma for level; default to be .5
    seasonal_knot_pooling_scale : float
        pooling sigma for seasonal fourier series regressors; default to be 1
    seasonal_knot_scale : float
        sigma for seasonal fourier series regressors; default to be 0.1.
    span_level : float between (0, 1)
        window width to decide the number of windows for the level (trend) term.
        e.g., span 0.1 will produce 10 windows.
    span_coefficients : float between (0, 1)
        window width to decide the number of windows for the regression term
    degree of freedom : int
        degree of freedom for error t-distribution
    level_knot_dates : array like
        list of pre-specified dates for the level knots
    level_knot_length : int
        the distance between every two knots for level
    coefficients_knot_length : int
        the distance between every two knots for coefficients
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.

    kwargs
        To specify `estimator_type` or additional args for the specified `estimator_type`

    """
    _data_input_mapper = constants.DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'ktrlite'

    def __init__(self,
                 seasonality=None,
                 seasonality_fs_order=None,
                 level_knot_scale=0.5,
                 seasonal_knot_pooling_scale=1.0,
                 seasonal_knot_scale=0.1,
                 span_level=0.1,
                 span_coefficients=0.3,
                 # rho_coefficients=0.15,
                 degree_of_freedom=30,
                 # knot customization
                 level_knot_dates=None,
                 level_knot_length=None,
                 coefficients_knot_length=None,
                 date_freq=None,
                 **kwargs):
        super().__init__(**kwargs)  # create estimator in base class

        self.span_level = span_level
        self.level_knot_scale = level_knot_scale
        # customize knot dates for levels
        self.level_knot_dates = level_knot_dates
        self.level_knot_length = level_knot_length
        self.coefficients_knot_length = coefficients_knot_length

        self.seasonality = seasonality
        self.seasonality_fs_order = seasonality_fs_order
        self.seasonal_knot_pooling_scale = seasonal_knot_pooling_scale
        self.seasonal_knot_scale = seasonal_knot_scale

        self.span_coefficients = span_coefficients
        # self.rho_coefficients = rho_coefficients
        self.date_freq = date_freq

        # set private var to arg value
        # if None set default in _set_default_base_args()
        # use public one if knots length is not available
        self._seasonality = self.seasonality
        self._seasonality_fs_order = self.seasonality_fs_order
        self._seasonal_knot_scale = self.seasonal_knot_scale
        self._seasonal_knot_pooling_scale = None
        self._seasonal_knot_scale = None

        self._level_knot_dates = self.level_knot_dates
        self._degree_of_freedom = degree_of_freedom

        self._training_df_meta = None
        self._model_data_input = dict()

        # regression attributes -- now is only used for fourier series as seasonality
        self._num_of_regressors = 0
        self._regressor_col = list()
        self._regressor_col_gp = list()
        self._coefficients_knot_pooling_loc = list()
        self._coefficients_knot_pooling_scale = list()
        self._coefficients_knot_scale = list()

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
        self._response_mean = 0
        self._is_valid_response = None
        self._which_valid_response = None
        self._num_of_valid_response = 0

        self._num_knots_level = None
        self._knots_tp_level = None

        self._num_knots_coefficients = None
        self._knots_tp_coefficients = None
        self._regressor_matrix = None
        self._coefficients_knot_dates = None

    def _set_default_base_args(self):
        """Set default attributes for None
        """
        if self.seasonality is None:
            self._seasonality = list()
            self._seasonality_fs_order = list()
        elif not isinstance(self._seasonality, list) and isinstance(self._seasonality * 1.0, float):
            self._seasonality = [self.seasonality]

        if self._seasonality and self._seasonality_fs_order is None:
            self._seasonality_fs_order = [2] * len(self._seasonality)
        elif not isinstance(self._seasonality_fs_order, list) and isinstance(self._seasonality_fs_order * 1.0, float):
            self._seasonality_fs_order = [self.seasonality_fs_order]

        if len(self._seasonality_fs_order) != len(self._seasonality):
            raise IllegalArgument('length of seasonality and fs_order not matching')

        for k, order in enumerate(self._seasonality_fs_order):
            if 2 * order > self._seasonality[k] - 1:
                raise IllegalArgument('reduce seasonality_fs_order to avoid over-fitting')

        if not isinstance(self.seasonal_knot_pooling_scale, list) and isinstance(self.seasonal_knot_pooling_scale * 1.0, float):
            self._seasonal_knot_pooling_scale = [self.seasonal_knot_pooling_scale] * len(self._seasonality)
        else:
            self._seasonal_knot_pooling_scale = self.seasonal_knot_pooling_scale

        if not isinstance(self.seasonal_knot_scale, list) and isinstance(self.seasonal_knot_scale * 1.0, float):
            self._seasonal_knot_scale = [self.seasonal_knot_scale] * len(self._seasonality)
        else:
            self._seasonal_knot_scale = self.seasonal_knot_scale

    def _set_seasonality_attributes(self):
        """given list of seasonalities and their order, create list of seasonal_regressors_columns"""
        self._regressor_col_gp = []
        self._regressor_col = []
        self._coefficients_knot_pooling_loc = []
        self._coefficients_knot_pooling_scale = []
        self._coefficients_knot_scale = []

        if len(self._seasonality) > 0:
            for idx, s in enumerate(self._seasonality):
                fs_cols = []
                order = self._seasonality_fs_order[idx]
                self._coefficients_knot_pooling_loc += [0.0] * order * 2
                self._coefficients_knot_pooling_scale += [self._seasonal_knot_pooling_scale[idx]] * order * 2
                self._coefficients_knot_scale += [self._seasonal_knot_scale[idx]] * order * 2
                for i in range(1, order + 1):
                    fs_cols.append('seas{}_fs_cos{}'.format(s, i))
                    fs_cols.append('seas{}_fs_sin{}'.format(s, i))
                # flatten version of regressor columns
                self._regressor_col += fs_cols
                # list of group of regressor columns bundled with seasonality
                self._regressor_col_gp.append(fs_cols)

        self._num_of_regressors = len(self._regressor_col)

    def _set_static_data_attributes(self):
        """model data input based on args at instantiation or computed from args at instantiation"""
        self._set_default_base_args()
        self._set_seasonality_attributes()

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

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ModelException("DataFrame does not contain `response_col`: {}".format(self.response_col))

        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            if self._num_of_observations < max_seasonality:
                raise ModelException(
                    "Number of observations {} is less than max seasonality {}".format(
                        self._num_of_observations, max_seasonality))

    def _make_seasonal_regressors(self, df, shift):
        """
        df : pd.DataFrame
        shift: int
            use 0 for fitting; use delta of prediction start and train start for prediction
        Returns
        -------
        pd.DataFrame
            data with computed fourier series attached
        """
        if len(self._seasonality) > 0:
            for idx, s in enumerate(self._seasonality):
                order = self._seasonality_fs_order[idx]
                df, _ = make_fourier_series_df(df, s, order=order, prefix='seas{}_'.format(s), shift=shift)

        return df

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
        self._regressor_matrix = np.zeros((self._num_of_observations, 0), dtype=np.double)
        if self._num_of_regressors > 0:
            self._regressor_matrix = df.filter(items=self._regressor_col,).values

    @staticmethod
    def get_gap_between_dates(start_date, end_date, freq):
        diff = end_date - start_date
        gap = np.array(diff / np.timedelta64(1, freq))

        return gap

    @staticmethod
    def set_knots_tp(knots_distance, cutoff):
        # start in the middle
        knots_idx_start = round(knots_distance / 2)
        knots_idx = np.arange(knots_idx_start, cutoff, knots_distance)

        return knots_idx

    def _set_kernel_matrix(self, df):
        # Note that our tp starts by 1; to convert back to index of array, reduce it by 1
        tp = np.arange(1, self._num_of_observations + 1) / self._num_of_observations

        # this approach put knots in full range
        self._cutoff = self._num_of_observations

        # kernel of level calculations
        if self._level_knot_dates is None:
            if self.level_knot_length is not None:
                # TODO: approximation; can consider level_knot_length directly as step size
                knots_distance = self.level_knot_length
            else:
                number_of_knots = round(1 / self.span_level)
                knots_distance = math.ceil(self._cutoff / number_of_knots)

            knots_idx_level = self.set_knots_tp(knots_distance, self._cutoff)
            self._knots_idx_level = knots_idx_level
            self._knots_tp_level = (1 + knots_idx_level) / self._num_of_observations
            self._level_knot_dates = df[self.date_col].values[knots_idx_level]
        else:
            self._level_knot_dates = pd.to_datetime([
                x for x in self._level_knot_dates if (x <= df[self.date_col].max()) \
                                                     and (x >= df[self.date_col].min())
            ])
            if self.date_freq is None:
                self.date_freq = pd.infer_freq(df[self.date_col])[0]
            start_date = self._training_df_meta['training_start']
            self._knots_tp_level = np.array(
                (self.get_gap_between_dates(start_date, self._level_knot_dates, self.date_freq) + 1) /
                (self.get_gap_between_dates(start_date, self._training_df_meta['training_end'], self.date_freq) + 1)
            )

        self._kernel_level = sandwich_kernel(tp, self._knots_tp_level)
        self._num_knots_level = len(self._knots_tp_level)

        self._kernel_coefficients = np.zeros((self._num_of_observations, 0), dtype=np.double)
        self._num_knots_coefficients = 0

        # kernel of coefficients calculations
        if self._num_of_regressors > 0:
            if self.coefficients_knot_length is not None:
                # TODO: approximation; can consider coefficients_knot_length directly as step size
                knots_distance = self.coefficients_knot_length
            else:
                number_of_knots = round(1 / self.span_coefficients)
                knots_distance = math.ceil(self._cutoff / number_of_knots)

            knots_idx_coef = self.set_knots_tp(knots_distance, self._cutoff)
            self._knots_idx_coef = knots_idx_coef
            self._knots_tp_coefficients = (1 + knots_idx_coef) / self._num_of_observations
            self._coef_knot_dates = df[self.date_col].values[knots_idx_coef]
            # self._kernel_coefficients = gauss_kernel(tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            self._kernel_coefficients = sandwich_kernel(tp, self._knots_tp_coefficients)
            self._num_knots_coefficients = len(self._knots_tp_coefficients)

    def _set_dynamic_data_attributes(self, df):
        """data input based on input DataFrame, rather than at object instantiation"""
        df = df.copy()
        self._response = df[self.response_col].values
        self._num_of_observations = len(self._response)

        self._validate_training_df(df)
        self._set_training_df_meta(df)

        df = self._make_seasonal_regressors(df, shift=0)
        self._set_valid_response_attributes()
        self._set_regressor_matrix(df)
        self._set_kernel_matrix(df)

    def _set_model_param_names(self):
        """Model parameters to extract"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]
        if len(self._seasonality) > 0 or self._num_of_regressors > 0:
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
                input_value = int(input_value)
            data_inputs[key.value] = input_value

        self._model_data_input = data_inputs

    def _get_model_data_input(self):
        return self._model_data_input

    def _predict(self, posterior_estimates, df, include_error=False, decompose=False):
        """Vectorized version of prediction math"""
        ################################################################
        # Model Attributes
        ################################################################

        model = deepcopy(posterior_estimates)
        # TODO: adopt torch as in lgt or dlt?
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

        if prediction_df_meta['prediction_start'] < training_df_meta['training_start']:
            raise PredictionException('Prediction start must be after training start.')

        trained_len = training_df_meta['df_length'] # i.e., self._num_of_observations
        output_len = prediction_df_meta['df_length']
        prediction_start = prediction_df_meta['prediction_start']

        # Here assume dates are ordered and consecutive
        # if prediction_df_meta['prediction_start'] > training_df_meta['training_end'],
        # assume prediction starts right after train end

        # If we cannot find a match of prediction range, assume prediction starts right after train end
        if prediction_start > training_df_meta['training_end']:
            # time index for prediction start
            start = trained_len
        else:
            start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

        df = self._make_seasonal_regressors(df, shift=start)
        new_tp = np.arange(start + 1, start + output_len + 1) / trained_len
        if include_error:
            # in-sample knots
            lev_knot_in = model.get(constants.BaseSamplingParameters.LEVEL_KNOT.value)
            # TODO: hacky way; let's just assume last two knot distance is knots distance for all knots
            lev_knot_width = self._knots_tp_level[-1] - self._knots_tp_level[-2]
            # check whether we need to put new knots for simulation
            if new_tp[-1] >= self._knots_tp_level[-1] + lev_knot_width:
                # derive knots tp
                knots_tp_level_out = np.arange(self._knots_tp_level[-1] + lev_knot_width, new_tp[-1], lev_knot_width)
                new_knots_tp_level = np.concatenate([self._knots_tp_level, knots_tp_level_out])
                lev_knot_out = np.random.laplace(0, self.level_knot_scale,
                                                 size=(lev_knot_in.shape[0], len(knots_tp_level_out)))
                lev_knot_out = np.cumsum(np.concatenate([lev_knot_in[:, -1].reshape(-1, 1), lev_knot_out],
                                                            axis=1), axis=1)[:, 1:]
                lev_knot = np.concatenate([lev_knot_in, lev_knot_out], axis=1)
            else:
                new_knots_tp_level = self._knots_tp_level
                lev_knot = lev_knot_in
            kernel_level = sandwich_kernel(new_tp, new_knots_tp_level)
        else:
            lev_knot = model.get(constants.BaseSamplingParameters.LEVEL_KNOT.value)
            kernel_level = sandwich_kernel(new_tp, self._knots_tp_level)

        obs_scale = model.get(constants.BaseSamplingParameters.OBS_SCALE.value)
        obs_scale = obs_scale.reshape(-1, 1)

        trend = np.matmul(lev_knot, kernel_level.transpose(1, 0))
        # init of regression matrix depends on length of response vector
        total_seas_regression = np.zeros(trend.shape, dtype=np.double)
        seas_decomp = {}
        # update seasonal regression matrices
        if self._seasonality and self._regressor_col:
            coef_knot = model.get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
            # kernel_coefficients = gauss_kernel(new_tp, self._knots_tp_coefficients, rho=self.rho_coefficients)
            kernel_coefficients = sandwich_kernel(new_tp, self._knots_tp_coefficients)
            coef = np.matmul(coef_knot, kernel_coefficients.transpose(1, 0))
            pos = 0
            for idx, cols in enumerate(self._regressor_col_gp):
                seasonal_regressor_matrix = df[cols].values
                seas_coef = coef[..., pos:(pos + len(cols)), :]
                seas_regression = np.sum(seas_coef * seasonal_regressor_matrix.transpose(1, 0), axis=-2)
                seas_decomp['seasonality_{}'.format(self._seasonality[idx])] = seas_regression
                pos += len(cols)
                total_seas_regression += seas_regression
        if include_error:
            epsilon = nct.rvs(self._degree_of_freedom, nc=0, loc=0,
                              scale=obs_scale, size=(num_sample, len(new_tp)))
            pred_array = trend + total_seas_regression + epsilon
        else:
            pred_array = trend + total_seas_regression

        # if decompose output dictionary of components
        if decompose:
            decomp_dict = {
                'prediction': pred_array,
                'trend': trend,
            }
            decomp_dict.update(seas_decomp)
        else:
            decomp_dict = {'prediction': pred_array}

        return decomp_dict

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


class KTRLiteMAP(BaseKTRLite, MAPTemplate):
    """Concrete KTRLite model for MAP (Maximum a Posteriori) prediction

    This model only supports MAP estimating `estimator_type`s
    """
    _supported_estimator_types = [StanEstimatorMAP]

    def __init__(self, estimator_type=StanEstimatorMAP, **kwargs):
        super().__init__(estimator_type=estimator_type, **kwargs)

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        super().fit(df)
        self._set_map_posterior()

    def predict(self, df, decompose=False, **kwargs):
        return self._map_predict(df, self._predict, decompose=decompose)
