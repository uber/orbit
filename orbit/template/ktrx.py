import pandas as pd
import numpy as np
import math
from scipy.stats import nct
from enum import Enum
import torch
from copy import deepcopy
from ..constants.constants import CoefPriorDictKeys

from ..exceptions import IllegalArgument, ModelException
from ..utils.general import is_ordered_datetime
from ..utils.kernels import gauss_kernel, sandwich_kernel
from ..utils.features import make_fourier_series_df
from .model_template import ModelTemplate


class DataInputMapper(Enum):
    """
    mapping from object input to pyro input
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    NUM_OF_OBSERVATIONS = 'N_OBS'
    RESPONSE = 'RESPONSE'
    NUM_OF_VALID_RESPONSE = 'N_VALID_RES'
    WHICH_VALID_RESPONSE = 'WHICH_VALID_RES'
    RESPONSE_SD = 'SDY'
    RESPONSE_OFFSET = 'MEAN_Y'
    DEGREE_OF_FREEDOM = 'DOF'
    MIN_RESIDUALS_SD = 'MIN_RESIDUALS_SD'
    # ----------  Level  ---------- #
    _NUM_KNOTS_LEVEL = 'N_KNOTS_LEV'
    LEVEL_KNOT_SCALE = 'LEV_KNOT_SCALE'
    _KERNEL_LEVEL = 'K_LEV'
    # ----------  Regression  ---------- #
    _NUM_KNOTS_COEFFICIENTS  = 'N_KNOTS_COEF'
    _KERNEL_COEFFICIENTS = 'K_COEF'
    _NUM_OF_REGULAR_REGRESSORS = 'N_RR'
    _NUM_OF_POSITIVE_REGRESSORS = 'N_PR'
    _REGULAR_REGRESSOR_MATRIX = 'RR'
    _POSITIVE_REGRESSOR_MATRIX = 'PR'
    _REGULAR_REGRESSOR_INIT_KNOT_LOC = 'RR_INIT_KNOT_LOC'
    _REGULAR_REGRESSOR_INIT_KNOT_SCALE = 'RR_INIT_KNOT_SCALE'
    _REGULAR_REGRESSOR_KNOT_SCALE = 'RR_KNOT_SCALE'
    _POSITIVE_REGRESSOR_INIT_KNOT_LOC = 'PR_INIT_KNOT_LOC'
    _POSITIVE_REGRESSOR_INIT_KNOT_SCALE = 'PR_INIT_KNOT_SCALE'
    _POSITIVE_REGRESSOR_KNOT_SCALE = 'PR_KNOT_SCALE'
    # ----------  Prior Specification  ---------- #
    _COEF_PRIOR_LIST = 'COEF_PRIOR_LIST'
    _LEVEL_KNOTS = 'LEV_KNOT_LOC'
    _SEAS_TERM = 'SEAS_TERM'
    # --------------- mvn
    MVN = 'MVN'
    GEOMETRIC_WALK = 'GEOMETRIC_WALK'


class BaseSamplingParameters(Enum):
    """
    The output sampling parameters related with the base model
    """
    LEVEL_KNOT = 'lev_knot'
    LEVEL = 'lev'
    YHAT = 'yhat'
    OBS_SCALE = 'obs_scale'


class RegressionSamplingParameters(Enum):
    """
    The output sampling parameters related with regression component.
    """
    COEFFICIENTS_KNOT = 'coef_knot'
    COEFFICIENTS_INIT_KNOT= 'coef_init_knot'
    COEFFICIENTS = 'coef'


# Defaults Values
DEFAULT_REGRESSOR_SIGN = '='
DEFAULT_COEFFICIENTS_INIT_KNOT_SCALE = 1.0
DEFAULT_COEFFICIENTS_INIT_KNOT_LOC = 0
DEFAULT_COEFFICIENTS_KNOT_SCALE = 0.1
DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER = 0.01
DEFAULT_UPPER_BOUND_SCALE_MULTIPLIER = 1.0


class BaseKTRX(ModelTemplate):
    """Base KTRX model object with shared functionality for PyroVI method
    Parameters
    ----------
    level_knot_scale : float
        sigma for level; default to be .1
    regressor_col : array-like strings
        regressor columns
    regressor_sign : list
        list of signs with '=' for regular regressor and '+' for positive regressor
    regressor_init_knot_loc : list
        list of regressor knot pooling mean priors, default to be 0's
    regressor_init_knot_scale : list
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
            '_seasonality' : seasonality order
            '_seasonality_fs_order' : fourier series order for seasonality
    coefficients_knot_length : int
        the distance between every two knots for coefficients
    coefficients_knot_dates : array like
        a list of pre-specified knot dates for coefficients
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    min_residuals_sd : float
        a numeric value from 0 to 1 to indicate the upper bound of residual scale parameter; e.g.
        0.5 means residual scale will be sampled from [0, 0.5] in a scaled Beta(2, 2) dist.
    flat_multiplier : bool
        Default set as True. If False, we will adjust knot scale with a multiplier based on regressor volume
        around each knot; When True, set all multiplier as 1
    geometric_walk : bool
        Default set as False. If True we will sample positive regressor knot as geometric random walk
    kwargs
        To specify `estimator_type` or additional args for the specified `estimator_type`
    """
    _data_input_mapper = DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = 'ktrx'

    def __init__(self,
                 level_knot_scale=0.1,
                 regressor_col=None,
                 regressor_sign=None,
                 regressor_init_knot_loc=None,
                 regressor_init_knot_scale=None,
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
                 mvn=0,
                 flat_multiplier=True,
                 geometric_walk=False,
                 min_residuals_sd=1.0,
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
        self.regressor_init_knot_loc = regressor_init_knot_loc
        self.regressor_init_knot_scale = regressor_init_knot_scale
        self.regressor_knot_scale = regressor_knot_scale

        self.coefficients_knot_length = coefficients_knot_length
        self.span_coefficients = span_coefficients
        self.rho_coefficients = rho_coefficients
        self.date_freq = date_freq

        self.degree_of_freedom = degree_of_freedom

        # multi var norm flag
        self.mvn = mvn
        # flat_multiplier flag
        self.flat_multiplier = flat_multiplier
        self.geometric_walk = geometric_walk
        self.min_residuals_sd = min_residuals_sd

        # set private var to arg value
        # if None set default in _set_default_args()
        self._regressor_sign = self.regressor_sign
        self._regressor_init_knot_loc = self.regressor_init_knot_loc
        self._regressor_init_knot_scale = self.regressor_init_knot_scale
        self._regressor_knot_scale = self.regressor_knot_scale

        self.coef_prior_list = coef_prior_list
        self._coef_prior_list = []
        self._coefficients_knot_dates = coefficients_knot_dates
        self._knots_idx_coef = None

        self._num_of_regressors = 0
        self._num_knots_coefficients = 0

        # positive regressors
        self._num_of_positive_regressors = 0
        self._positive_regressor_col = list()
        self._positive_regressor_init_knot_loc = list()
        self._positive_regressor_init_knot_scale = list()
        self._positive_regressor_knot_scale = list()
        # regular regressors
        self._num_of_regular_regressors = 0
        self._regular_regressor_col = list()
        self._regular_regressor_init_knot_loc = list()
        self._regular_regressor_init_knot_scale = list()
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

        self._set_static_attributes()

    def _set_model_param_names(self):
        """Overriding base template functions. Model parameters to extract"""
        self._model_param_names += [param.value for param in BaseSamplingParameters]
        if self._num_of_regressors > 0:
            self._model_param_names += [param.value for param in RegressionSamplingParameters]

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
            self._regressor_init_knot_loc = list()
            self._regressor_init_knot_scale = list()
            self._regressor_knot_scale = list()

            return

        def _validate_params_len(params, valid_length):
            for p in params:
                if p is not None and len(p) != valid_length:
                    raise IllegalArgument('Wrong dimension length in Regression Param Input')

        # regressor defaults
        num_of_regressors = len(self.regressor_col)

        _validate_params_len([
            self.regressor_sign, self.regressor_init_knot_loc,
            self.regressor_init_knot_scale, self.regressor_knot_scale],
            num_of_regressors
        )

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * num_of_regressors

        if self.regressor_init_knot_loc is None:
            self._regressor_init_knot_loc = [DEFAULT_COEFFICIENTS_INIT_KNOT_LOC] * num_of_regressors

        if self.regressor_init_knot_scale is None:
            self._regressor_init_knot_scale = [DEFAULT_COEFFICIENTS_INIT_KNOT_SCALE] * num_of_regressors

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
                self._positive_regressor_init_knot_loc.append(self._regressor_init_knot_loc[index])
                self._positive_regressor_init_knot_scale.append(self._regressor_init_knot_scale[index])
                # used for 'pr_knot' sampling in pyro
                self._positive_regressor_knot_scale.append(self._regressor_knot_scale[index])
            else:
                self._num_of_regular_regressors += 1
                self._regular_regressor_col.append(self.regressor_col[index])
                # used for 'rr_knot_loc' sampling in pyro
                self._regular_regressor_init_knot_loc.append(self._regressor_init_knot_loc[index])
                self._regular_regressor_init_knot_scale.append(self._regressor_init_knot_scale[index])
                # used for 'rr_knot' sampling in pyro
                self._regular_regressor_knot_scale.append(self._regressor_knot_scale[index])
        # regular first, then positive
        self._regressor_col = self._regular_regressor_col + self._positive_regressor_col
        # numpy conversion
        self._positive_regressor_init_knot_loc = np.array(self._positive_regressor_init_knot_loc)
        self._positive_regressor_init_knot_scale = np.array(self._positive_regressor_init_knot_scale)
        self._positive_regressor_knot_scale = np.array(self._positive_regressor_knot_scale)
        self._regular_regressor_init_knot_loc = np.array(self._regular_regressor_init_knot_loc)
        self._regular_regressor_init_knot_scale = np.array(self._regular_regressor_init_knot_scale)
        self._regular_regressor_knot_scale = np.array(self._regular_regressor_knot_scale)

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
        """provide a array like outcome of index based on the knots distance and cutoff point"""
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

    def _set_valid_response_attributes(self, training_meta):
        num_of_observations = training_meta['num_of_observations']
        response = training_meta['response']

        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            if num_of_observations < max_seasonality:
                raise ModelException(
                    "Number of observations {} is less than max seasonality {}".format(
                        num_of_observations, max_seasonality))
        # get some reasonable offset to regularize response to make default priors scale-insensitive
        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            self.response_offset = np.nanmean(response[:max_seasonality])
        else:
            self.response_offset = np.nanmean(response)

        self.is_valid_response = ~np.isnan(response)
        # [0] to convert tuple back to array
        self.which_valid_response = np.where(self.is_valid_response)[0]
        self.num_of_valid_response = len(self.which_valid_response)

    def _set_regressor_matrix(self, df, training_meta):
        num_of_observations = training_meta['num_of_observations']
        # validate regression columns
        if self.regressor_col is not None and \
                not set(self.regressor_col).issubset(df.columns):
            raise ModelException(
                "DataFrame does not contain specified regressor column(s)."
            )

        # init of regression matrix depends on length of response vector
        self._positive_regressor_matrix = np.zeros((num_of_observations, 0), dtype=np.double)
        self._regular_regressor_matrix = np.zeros((num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self._num_of_positive_regressors > 0:
            self._positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col,).values

        if self._num_of_regular_regressors > 0:
            self._regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col,).values

    def _set_coefficients_kernel_matrix(self, df, training_meta):
        """Derive knots position and kernel matrix and other related meta data"""
        num_of_observations = training_meta['num_of_observations']
        date_col = training_meta['date_col']
        # Note that our tp starts by 1; to convert back to index of array, reduce it by 1
        tp = np.arange(1, num_of_observations + 1) / num_of_observations
        # this approach put knots in full range
        # TODO: consider deprecate _cutoff for now since we assume _cutoff always the same as num of obs?
        self._cutoff = num_of_observations
        self._kernel_coefficients = np.zeros((num_of_observations, 0), dtype=np.double)
        self._num_knots_coefficients = 0

        # kernel of coefficients calculations
        if self._num_of_regressors > 0:
            # if users didn't provide knot positions, evenly distribute it based on span_coefficients
            # or knot length provided by users
            if self._coefficients_knot_dates is None:
                # original code
                if self.coefficients_knot_length is not None:
                    knots_distance = self.coefficients_knot_length
                else:
                    number_of_knots = round(1 / self.span_coefficients)
                    knots_distance = math.ceil(self._cutoff / number_of_knots)
                # derive actual date arrays based on the time-point (tp) index
                knots_idx_coef = self._set_knots_tp(knots_distance, self._cutoff)
                self._knots_tp_coefficients = (1 + knots_idx_coef) / num_of_observations
                self._coefficients_knot_dates = df[date_col].values[knots_idx_coef]
                self._knots_idx_coef = knots_idx_coef
                # TODO: new idea
                # # ignore this case for now
                # # if self.coefficients_knot_length is not None:
                # #     knots_distance = self.coefficients_knot_length
                # # else:
                # number_of_knots = round(1 / self.span_coefficients)
                # # to work with index; has to be discrete
                # knots_distance = math.ceil(self._cutoff / number_of_knots)
                # # always has a knot at the starting point
                # # derive actual date arrays based on the time-point (tp) index
                # knots_idx_coef = np.arange(0, self._cutoff, knots_distance)
                # self._knots_tp_coefficients = (1 + knots_idx_coef) / self.num_of_observations
                # self._coefficients_knot_dates = df[self.date_col].values[knots_idx_coef]
                # self._knots_idx_coef = knots_idx_coef
            else:
                # FIXME: this only works up to daily series (not working on hourly series)
                # FIXME: didn't provide  self.knots_idx_coef in this case
                self._coefficients_knot_dates = pd.to_datetime([
                    x for x in self._coefficients_knot_dates if (x <= df[date_col].max()) \
                                                                and (x >= df[date_col].min())
                ])
                if self.date_freq is None:
                    self.date_freq = pd.infer_freq(df[date_col])[0]
                start_date = training_meta['training_start']
                end_date = training_meta['training_end']
                self._knots_idx_coef = (
                    self._get_gap_between_dates(start_date, self._coefficients_knot_dates, self.date_freq)
                )

                self._knots_tp_coefficients = np.array(
                    (self._knots_idx_coef + 1) /
                    (self._get_gap_between_dates(start_date, end_date, self.date_freq) + 1)
                )
                self._knots_idx_coef = list(self._knots_idx_coef.astype(np.int32))

            kernel_coefficients = gauss_kernel(tp, self._knots_tp_coefficients, rho=self.rho_coefficients)

            self._num_knots_coefficients = len(self._knots_tp_coefficients)
            self._kernel_coefficients = kernel_coefficients

    def _set_knots_scale_matrix(self, df, training_meta):
        num_of_observations = training_meta['num_of_observations']
        if self._num_of_positive_regressors > 0:
            # calculate average local absolute volume for each segment
            local_val = np.ones((self._num_of_positive_regressors, self._num_knots_coefficients))
            if self.flat_multiplier:
                multiplier = np.ones(local_val.shape)
            else:
                multiplier = np.ones(local_val.shape)
                # store local value for the range on the left side since last knot
                for idx in range(len(self._knots_idx_coef)):
                    if idx < len(self._knots_idx_coef) - 1:
                        str_idx = self._knots_idx_coef[idx]
                        end_idx = self._knots_idx_coef[idx + 1]
                    else:
                        str_idx = self._knots_idx_coef[idx]
                        end_idx = num_of_observations

                    local_val[:, idx] = np.mean(np.fabs(self._positive_regressor_matrix[str_idx:end_idx]), axis=0)

                # adjust knot scale with the multiplier derive by the average value and shift by 0.001 to avoid zeros in
                # scale parameters
                global_med = np.expand_dims(np.mean(np.fabs(self._positive_regressor_matrix), axis=0), -1)
                test_flag = local_val < 0.01 * global_med

                multiplier[test_flag] = DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER
                # replace entire row of nan (when 0.1 * global_med is equal to global_min) with upper bound
                multiplier[np.isnan(multiplier).all(axis=-1)] = 1.0

            # also note that after the following step,
            # _positive_regressor_knot_scale is a 2D array unlike _regular_regressor_knot_scale
            # geometric drift i.e. 0.1 = 10% up-down in 1 s.d. prob.
            # after line below, self._positive_regressor_knot_scale has shape num_of_pr x num_of_knot
            self._positive_regressor_knot_scale = (
                    multiplier * np.expand_dims(self._positive_regressor_knot_scale, -1)
            )
            # keep a lower bound of scale parameters
            self._positive_regressor_knot_scale[self._positive_regressor_knot_scale < 1e-4] = 1e-4
            # TODO: we change the type here, maybe we should change it earlier?
            self._positive_regressor_init_knot_scale = np.array(self._positive_regressor_init_knot_scale)
            self._positive_regressor_init_knot_scale[self._positive_regressor_init_knot_scale < 1e-4] = 1e-4

        if self._num_of_regular_regressors > 0:
            # do the same for regular regressor
            # calculate average local absolute volume for each segment
            local_val = np.ones((self._num_of_regular_regressors, self._num_knots_coefficients))
            if self.flat_multiplier:
                multiplier = np.ones(local_val.shape)
            else:
                multiplier = np.ones(local_val.shape)
            # store local value for the range on the left side since last knot
            for idx in range(len(self._knots_idx_coef)):
                if idx < len(self._knots_idx_coef) - 1:
                    str_idx = self._knots_idx_coef[idx]
                    end_idx = self._knots_idx_coef[idx + 1]
                else:
                    str_idx = self._knots_idx_coef[idx]
                    end_idx = num_of_observations

                local_val[:, idx] = np.mean(np.fabs(self._regular_regressor_matrix[str_idx:end_idx]), axis=0)

            # adjust knot scale with the multiplier derive by the average value and shift by 0.001 to avoid zeros in
            # scale parameters
            global_med = np.expand_dims(np.median(np.fabs(self._regular_regressor_matrix), axis=0), -1)
            test_flag = local_val < 0.01 * global_med
            multiplier[test_flag] = DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER
            # replace entire row of nan (when 0.1 * global_med is equal to global_min) with upper bound
            multiplier[np.isnan(multiplier).all(axis=-1)] = 1.0

            # also note that after the following step,
            # _regular_regressor_knot_scale is a 2D array unlike _regular_regressor_knot_scale
            # geometric drift i.e. 0.1 = 10% up-down in 1 s.d. prob.
            # self._regular_regressor_knot_scale has shape num_of_pr x num_of_knot
            self._regular_regressor_knot_scale = (
                    multiplier * np.expand_dims(self._regular_regressor_knot_scale, -1)
            )
            # keep a lower bound of scale parameters
            self._regular_regressor_knot_scale[self._regular_regressor_knot_scale < 1e-4] = 1e-4
            # TODO: we change the type here, maybe we should change it earlier?
            self._regular_regressor_init_knot_scale = np.array(self._regular_regressor_init_knot_scale)
            self._regular_regressor_init_knot_scale[self._regular_regressor_init_knot_scale < 1e-4] = 1e-4

    def _generate_tp(self, training_meta, prediction_date_array):
        """Used in _generate_coefs"""
        training_end = training_meta['training_end']
        num_of_observations = training_meta['num_of_observations']
        date_array = training_meta['date_array']
        prediction_start = prediction_date_array[0]
        output_len = len(prediction_date_array)
        if prediction_start > training_end:
            start = num_of_observations
        else:
            start = pd.Index(date_array).get_loc(prediction_start)

        new_tp = np.arange(start + 1, start + output_len + 1) / num_of_observations
        return new_tp

    def _generate_insample_tp(self, training_meta, date_array):
        """Used in _generate_coefs"""
        train_date_array = training_meta['date_array']
        num_of_observations = training_meta['num_of_observations']
        idx = np.nonzero(np.in1d(train_date_array, date_array))[0]
        tp = (idx + 1) / num_of_observations
        return tp

    def _generate_coefs(self, training_meta, prediction_date_array, coef_knot_dates, coef_knot):
        """Used in _generate_seas"""
        new_tp = self._generate_tp(training_meta, prediction_date_array)
        knots_tp_coef = self._generate_insample_tp(training_meta, coef_knot_dates)
        kernel_coef = sandwich_kernel(new_tp, knots_tp_coef)
        coefs = np.squeeze(np.matmul(coef_knot, kernel_coef.transpose(1, 0)), axis=0).transpose(1, 0)
        return coefs

    def _generate_seas(self, df, training_meta, coef_knot_dates, coef_knot, seasonality, seasonality_fs_order):
        """To calculate the seasonality term based on the _seasonal_knots_input.
        :param df: input df
        :param coef_knot_dates: dates for coef knots
        :param coef_knot: knot values for coef
        :param seasonality: seasonality input
        :param seasonality_fs_order: seasonality_fs_order input
        :return:
        """
        date_col = training_meta['date_col']
        date_array = training_meta['date_array']
        training_end = training_meta['training_end']
        num_of_observations = training_meta['num_of_observations']

        prediction_date_array = df[date_col].values
        prediction_start = prediction_date_array[0]

        df = df.copy()
        if prediction_start > training_end:
            forecast_dates = set(prediction_date_array)
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = num_of_observations
        else:
            # compute how many steps to forecast
            forecast_dates = set(prediction_date_array) - set(date_array)
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or \
                               - (len(set(date_array) - set(prediction_date_array)))
            # time index for prediction start
            start = pd.Index(date_array).get_loc(prediction_start)

        fs_cols = []
        for idx, s in enumerate(seasonality):
            order = seasonality_fs_order[idx]
            df, fs_cols_temp = make_fourier_series_df(df, s, order=order, prefix='seas{}_'.format(s), shift=start)
            fs_cols += fs_cols_temp

        sea_regressor_matrix = df.filter(items=fs_cols).values
        sea_coefs = self._generate_coefs(training_meta, prediction_date_array, coef_knot_dates, coef_knot)
        seas = np.sum(sea_coefs * sea_regressor_matrix, axis=-1)

        return seas

    def _set_levs_and_seas(self, df, training_meta):
        num_of_observations = training_meta['num_of_observations']
        date_col = training_meta['date_col']
        training_start = training_meta['training_start']
        training_end = training_meta['training_end']
        tp = np.arange(1, num_of_observations + 1) / num_of_observations
        # trim level knots dates when they are beyond training dates
        lev_knot_dates = list()
        lev_knots = list()
        # TODO: any faster way instead of a simple loop?
        for i, x in enumerate(self.level_knot_dates):
            if (x <= df[date_col].max()) and (x >= df[date_col].min()):
                lev_knot_dates.append(x)
                lev_knots.append(self.level_knots[i])
        self._level_knot_dates = pd.to_datetime(lev_knot_dates)
        self._level_knots = np.array(lev_knots)
        infer_freq = pd.infer_freq(df[date_col])[0]
        start_date = training_start

        if len(self.level_knots) > 0 and len(self.level_knot_dates) > 0:
            self.knots_tp_level = np.array(
                (self._get_gap_between_dates(start_date, self._level_knot_dates, infer_freq) + 1) /
                (self._get_gap_between_dates(start_date, training_end, infer_freq) + 1)
            )
        else:
            raise ModelException("User need to supply a list of level knots.")

        kernel_level = sandwich_kernel(tp, self.knots_tp_level)
        self._kernel_level = kernel_level
        self._num_knots_level = len(self._level_knot_dates)

        if self._seasonal_knots_input is not None:
            self._seas_term = self._generate_seas(
                df,
                training_meta,
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

    def set_dynamic_attributes(self, df, training_meta):
        """Overriding: func: `~orbit.models.BaseETS._set_dynamic_attributes"""
        self._set_valid_response_attributes(training_meta)
        self._set_regressor_matrix(df, training_meta)
        self._set_coefficients_kernel_matrix(df)
        self._set_knots_scale_matrix()
        self._set_levs_and_seas(df, training_meta)
        self._filter_coef_prior(df)

    @staticmethod
    def _concat_regression_coefs(pr_beta=None, rr_beta=None):
        """Concatenates regression posterior matrix
        In the case that `pr_beta` or `rr_beta` is a 1d tensor, transform to 2d tensor and
        concatenate.
        Args
        ----
        pr_beta : array like
            postive-value constrainted regression betas
        rr_beta : array like
            regular regression betas
        Returns
        -------
        array like
            concatenated 2d array of shape (1, len(rr_beta) + len(pr_beta))
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

