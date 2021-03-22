import pandas as pd
import numpy as np
import math
from scipy.stats import nct
import torch
from copy import deepcopy

from ..constants import ktrlite as constants

from ..estimators.stan_estimator import StanEstimatorMAP
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..utils.general import is_ordered_datetime
from ..utils.kernels import sandwich_kernel
from ..utils.features import make_fourier_series_df
from .template import BaseTemplate, MAPTemplate


class BaseKTRLite(BaseTemplate):
    """Base KTRLite model object with shared functionality for MAP method

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
    # data labels for sampler
    _data_input_mapper = constants.DataInputMapper
    # used to match name of `*.stan` or `*.pyro` file to look for the model
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

        # set private var to arg value
        # if None set default in _set_default_args()
        # use public one if knots length is not available
        self._seasonality = self.seasonality
        self._seasonality_fs_order = self.seasonality_fs_order
        self._seasonal_knot_scale = self.seasonal_knot_scale
        self._seasonal_knot_pooling_scale = None
        self._seasonal_knot_scale = None

        self._level_knot_dates = self.level_knot_dates
        self._degree_of_freedom = degree_of_freedom

        self.span_coefficients = span_coefficients
        # self.rho_coefficients = rho_coefficients
        self.date_freq = date_freq

        # regression attributes -- now is ONLY used for fourier series as seasonality
        self.num_of_regressors = 0
        self.regressor_col = list()
        self.regressor_col_gp = list()
        self.coefficients_knot_pooling_loc = list()
        self.coefficients_knot_pooling_scale = list()
        self.coefficients_knot_scale = list()

        # set static data attributes
        # self._set_static_attributes()

        # set model param names
        # this only depends on static attributes, but should these params depend
        # on dynamic data (e.g actual data matrix, number of responses, etc) this should be
        # called after fit instead
        # self._set_model_param_names()

        # basic response fields
        # mainly set by ._set_dynamic_attributes()
        self.response_offset = 0
        self.is_valid_response = None
        self.which_valid_response = None
        self.num_of_valid_response = 0

        self.num_knots_level = None
        self.knots_tp_level = None

        self.num_knots_coefficients = None
        self.knots_tp_coefficients = None
        self.regressor_matrix = None
        # self.coefficients_knot_dates = None

    # initialization related modules
    def _set_default_args(self):
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

        if not isinstance(self.seasonal_knot_pooling_scale, list) and \
                isinstance(self.seasonal_knot_pooling_scale * 1.0, float):
            self._seasonal_knot_pooling_scale = [self.seasonal_knot_pooling_scale] * len(self._seasonality)
        else:
            self._seasonal_knot_pooling_scale = self.seasonal_knot_pooling_scale

        if not isinstance(self.seasonal_knot_scale, list) and isinstance(self.seasonal_knot_scale * 1.0, float):
            self._seasonal_knot_scale = [self.seasonal_knot_scale] * len(self._seasonality)
        else:
            self._seasonal_knot_scale = self.seasonal_knot_scale

    def _set_seasonality_attributes(self):
        """given list of seasonalities and their order, create list of seasonal_regressors_columns"""
        self.regressor_col_gp = list()
        self.regressor_col = list()
        self.coefficients_knot_pooling_loc = list()
        self.coefficients_knot_pooling_scale = list()
        self.coefficients_knot_scale = list()

        if len(self._seasonality) > 0:
            for idx, s in enumerate(self._seasonality):
                fs_cols = []
                order = self._seasonality_fs_order[idx]
                self.coefficients_knot_pooling_loc += [0.0] * order * 2
                self.coefficients_knot_pooling_scale += [self._seasonal_knot_pooling_scale[idx]] * order * 2
                self.coefficients_knot_scale += [self._seasonal_knot_scale[idx]] * order * 2
                for i in range(1, order + 1):
                    fs_cols.append('seas{}_fs_cos{}'.format(s, i))
                    fs_cols.append('seas{}_fs_sin{}'.format(s, i))
                # flatten version of regressor columns
                self.regressor_col += fs_cols
                # list of group of regressor columns bundled with seasonality
                self.regressor_col_gp.append(fs_cols)

        self.num_of_regressors = len(self.regressor_col)

    def _set_static_attributes(self):
        """Over-ride function from Base Template"""
        self._set_default_args()
        self._set_seasonality_attributes()

    # fit and predict related modules
    def _set_validate_ktr_params(self, df):
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

    def _set_regressor_matrix(self, df):
        # init of regression matrix depends on length of response vector
        self.regressor_matrix = np.zeros((self.num_of_observations, 0), dtype=np.double)
        if self.num_of_regressors > 0:
            self.regressor_matrix = df.filter(items=self.regressor_col, ).values

    @staticmethod
    def get_gap_between_dates(start_date, end_date, freq):
        diff = end_date - start_date
        gap = np.array(diff / np.timedelta64(1, freq))

        return gap

    @staticmethod
    def _set_knots_tp(knots_distance, cutoff):
        # start in the middle
        knots_idx_start = round(knots_distance / 2)
        knots_idx = np.arange(knots_idx_start, cutoff, knots_distance)

        return knots_idx

    def _set_kernel_matrix(self, df):
        # Note that our tp starts by 1; to convert back to index of array, reduce it by 1
        tp = np.arange(1, self.num_of_observations + 1) / self.num_of_observations

        # this approach put knots in full range
        self._cutoff = self.num_of_observations

        # kernel of level calculations
        if self._level_knot_dates is None:
            if self.level_knot_length is not None:
                # TODO: approximation; can consider level_knot_length directly as step size
                knots_distance = self.level_knot_length
            else:
                number_of_knots = round(1 / self.span_level)
                knots_distance = math.ceil(self._cutoff / number_of_knots)

            knots_idx_level = self._set_knots_tp(knots_distance, self._cutoff)
            self._knots_idx_level = knots_idx_level
            self.knots_tp_level = (1 + knots_idx_level) / self.num_of_observations
            self._level_knot_dates = df[self.date_col].values[knots_idx_level]
        else:
            self._level_knot_dates = pd.to_datetime([
                x for x in self._level_knot_dates if 
                (x <= df[self.date_col].max()) and (x >= df[self.date_col].min())
            ])
            if self.date_freq is None:
                self.date_freq = pd.infer_freq(df[self.date_col])[0]
            start_date = self.training_start
            self.knots_tp_level = np.array(
                (self.get_gap_between_dates(start_date, self._level_knot_dates, self.date_freq) + 1) /
                (self.get_gap_between_dates(start_date, self.training_end, self.date_freq) + 1)
            )

        self.kernel_level = sandwich_kernel(tp, self.knots_tp_level)
        self.num_knots_level = len(self.knots_tp_level)

        self.kernel_coefficients = np.zeros((self.num_of_observations, 0), dtype=np.double)
        self.num_knots_coefficients = 0

        # kernel of coefficients calculations
        if self.num_of_regressors > 0:
            if self.coefficients_knot_length is not None:
                # TODO: approximation; can consider coefficients_knot_length directly as step size
                knots_distance = self.coefficients_knot_length
            else:
                number_of_knots = round(1 / self.span_coefficients)
                knots_distance = math.ceil(self._cutoff / number_of_knots)

            knots_idx_coef = self._set_knots_tp(knots_distance, self._cutoff)
            self._knots_idx_coef = knots_idx_coef
            self.knots_tp_coefficients = (1 + knots_idx_coef) / self.num_of_observations
            self._coef_knot_dates = df[self.date_col].values[knots_idx_coef]
            # self.kernel_coefficients = gauss_kernel(tp, self.knots_tp_coefficients, rho=self.rho_coefficients)
            self.kernel_coefficients = sandwich_kernel(tp, self.knots_tp_coefficients)
            self.num_knots_coefficients = len(self.knots_tp_coefficients)

    def _set_dynamic_attributes(self, df):
        """Overriding: func: `~orbit.models.BaseETS._set_dynamic_attributes"""
        # extra settings and validation for KTRLite
        self._set_validate_ktr_params(df)
        # attach fourier series as regressors
        df = self._make_seasonal_regressors(df, shift=0)
        # set regressors as input matrix and derive kernels
        self._set_regressor_matrix(df)
        self._set_kernel_matrix(df)

    def _set_model_param_names(self):
        """Model parameters to extract"""
        self._model_param_names += [param.value for param in constants.BaseSamplingParameters]
        if len(self._seasonality) > 0 or self.num_of_regressors > 0:
            self._model_param_names += [param.value for param in constants.RegressionSamplingParameters]

    def _predict(self, posterior_estimates, df, include_error=False, decompose=False, **kwargs):
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

        if prediction_df_meta['prediction_start'] < self.training_start:
            raise PredictionException('Prediction start must be after training start.')

        trained_len = self.num_of_observations  # i.e., self.num_of_observations
        output_len = prediction_df_meta['df_length']
        prediction_start = prediction_df_meta['prediction_start']

        # Here assume dates are ordered and consecutive
        # if prediction_df_meta['prediction_start'] > training_df_meta['training_end'],
        # assume prediction starts right after train end

        # If we cannot find a match of prediction range, assume prediction starts right after train end
        if prediction_start > self.training_end:
            # time index for prediction start
            start = trained_len
        else:
            start = pd.Index(self.date_array).get_loc(prediction_start)

        df = self._make_seasonal_regressors(df, shift=start)
        new_tp = np.arange(start + 1, start + output_len + 1) / trained_len
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

        trend = np.matmul(lev_knot, kernel_level.transpose(1, 0))
        # init of regression matrix depends on length of response vector
        total_seas_regression = np.zeros(trend.shape, dtype=np.double)
        seas_decomp = {}
        # update seasonal regression matrices
        if self._seasonality and self.regressor_col:
            coef_knot = model.get(constants.RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
            # kernel_coefficients = gauss_kernel(new_tp, self.knots_tp_coefficients, rho=self.rho_coefficients)
            kernel_coefficients = sandwich_kernel(new_tp, self.knots_tp_coefficients)
            coef = np.matmul(coef_knot, kernel_coefficients.transpose(1, 0))
            pos = 0
            for idx, cols in enumerate(self.regressor_col_gp):
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


class KTRLiteMAP(MAPTemplate, BaseKTRLite):
    """Concrete KTRLite model for MAP (Maximum a Posteriori) prediction

    This model only supports MAP estimating `estimator_type`s
    """
    _supported_estimator_types = [StanEstimatorMAP]

    def __init__(self, estimator_type=StanEstimatorMAP, **kwargs):
        super().__init__(estimator_type=estimator_type, **kwargs)
