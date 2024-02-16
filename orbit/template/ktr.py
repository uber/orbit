import pandas as pd
import numpy as np
import math
from scipy.stats import nct
from enum import Enum
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from ..constants.constants import (
    KTRTimePointPriorKeys,
    PredictMethod,
    TrainingMetaKeys,
    PredictionMetaKeys,
)
from ..constants.palette import OrbitPalette
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..utils.general import is_ordered_datetime
from ..utils.kernels import gauss_kernel, sandwich_kernel
from ..utils.features import make_seasonal_regressors
from .model_template import ModelTemplate
from ..estimators.pyro_estimator import PyroEstimatorSVI
from ..models import KTRLite
from ..utils.knots import get_knot_idx, get_knot_dates
from ..utils.plot import orbit_style_decorator


class DataInputMapper(Enum):
    """
    mapping from object input to pyro input
    """

    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    NUM_OF_VALID_RESPONSE = "N_VALID_RES"
    WHICH_VALID_RESPONSE = "WHICH_VALID_RES"
    RESPONSE_OFFSET = "MEAN_Y"
    DEGREE_OF_FREEDOM = "DOF"
    _RESIDUALS_SCALE_UPPER = "RESID_SCALE_UB"
    # ----------  Level  ---------- #
    _NUM_KNOTS_LEVEL = "N_KNOTS_LEV"
    LEVEL_KNOT_SCALE = "LEV_KNOT_SCALE"
    _KERNEL_LEVEL = "K_LEV"
    # ----------  Regression  ---------- #
    _NUM_KNOTS_COEFFICIENTS = "N_KNOTS_COEF"
    _KERNEL_COEFFICIENTS = "K_COEF"
    _NUM_OF_REGULAR_REGRESSORS = "N_RR"
    _NUM_OF_POSITIVE_REGRESSORS = "N_PR"
    _NUM_OF_NEGATIVE_REGRESSORS = "N_NR"
    _REGULAR_REGRESSOR_MATRIX = "RR"
    _POSITIVE_REGRESSOR_MATRIX = "PR"
    _NEGATIVE_REGRESSOR_MATRIX = "NR"
    _REGULAR_REGRESSOR_INIT_KNOT_LOC = "RR_INIT_KNOT_LOC"
    _REGULAR_REGRESSOR_INIT_KNOT_SCALE = "RR_INIT_KNOT_SCALE"
    _REGULAR_REGRESSOR_KNOT_SCALE = "RR_KNOT_SCALE"
    _POSITIVE_REGRESSOR_INIT_KNOT_LOC = "PR_INIT_KNOT_LOC"
    _POSITIVE_REGRESSOR_INIT_KNOT_SCALE = "PR_INIT_KNOT_SCALE"
    _POSITIVE_REGRESSOR_KNOT_SCALE = "PR_KNOT_SCALE"
    _NEGATIVE_REGRESSOR_INIT_KNOT_LOC = "NR_INIT_KNOT_LOC"
    _NEGATIVE_REGRESSOR_INIT_KNOT_SCALE = "NR_INIT_KNOT_SCALE"
    _NEGATIVE_REGRESSOR_KNOT_SCALE = "NR_KNOT_SCALE"
    # ----------  Prior Specification  ---------- #
    _COEF_PRIOR_LIST = "COEF_PRIOR_LIST"
    _LEVEL_KNOTS = "LEV_KNOT_LOC"
    _SEAS_TERM = "SEAS_TERM"


class BaseSamplingParameters(Enum):
    """
    The output sampling parameters related with the base model
    """

    LEVEL_KNOT = "lev_knot"
    LEVEL = "lev"
    YHAT = "yhat"
    OBS_SCALE = "obs_scale"


class RegressionSamplingParameters(Enum):
    """
    The output sampling parameters related with regression component.
    """

    COEFFICIENTS_KNOT = "coef_knot"
    COEFFICIENTS_INIT_KNOT = "coef_init_knot"
    COEFFICIENTS = "coef"


# Defaults Values
DEFAULT_REGRESSOR_SIGN = "="
DEFAULT_COEFFICIENTS_INIT_KNOT_SCALE = 1.0
DEFAULT_COEFFICIENTS_INIT_KNOT_LOC = 0
DEFAULT_COEFFICIENTS_KNOT_SCALE = 0.1
DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER = 0.01
DEFAULT_UPPER_BOUND_SCALE_MULTIPLIER = 1.0


class KTRModel(ModelTemplate):
    """Base KTR model object with shared functionality for PyroVI method
    Parameters
    ----------
    level_knot_scale : float
        sigma for level; default to be .1
    level_segments : int
        the number of segments partitioned by the knots of level (trend)
    level_knot_distance : int
        the distance between every two knots of level (trend)
    level_knot_dates : array like
        list of pre-specified dates for the level knots
    seasonality : int, or list of int
        multiple seasonality
    seasonality_fs_order : int, or list of int
        fourier series order for seasonality
    seasonality_segments : int
        the number of segments partitioned by the knots of seasonality
    seasonal_initial_knot_scale : float
        scale parameter for seasonal regressors initial coefficient knots; default to be 1
    seasonal_knot_scale : float
        scale parameter for seasonal regressors drift of coefficient knots; default to be 0.1.
    regressor_col : array-like strings
        regressor columns
    regressor_sign : list
        list of signs with '=' for regular regressor, '+' for positive regressor, and '-' for negative regressor.
    regressor_init_knot_loc : list
        list of regressor knot pooling mean priors, default to be 0's
    regressor_init_knot_scale : list
        list of regressor knot pooling sigma's to control the pooling strength towards the grand mean of regressors;
        default to be 1.
    regressor_knot_scale : list
        list of regressor knot sigma priors; default to be 0.1.
    regression_segments : int
        the number of segments partitioned by the knots of regression
    regression_knot_distance : int
        the distance between every two knots of regression
    regression_knot_dates : array-like
        list of pre-specified dates for regression knots
    regression_rho : float
        sigma in the Gaussian kernel for the regression term
    degree of freedom : int
        degree of freedom for error t-distribution
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    coef_prior_list : list of dicts
        each dict in the list should have keys as
        'name', prior_start_tp_idx' (inclusive), KTRTimePointPriorKeys.PRIOR_END_TP_IDX.value (not inclusive),
        KTRTimePointPriorKeys.PRIOR_MEAN.value, KTRTimePointPriorKeys.PRIOR_SD.value, and KTRTimePointPriorKeys.PRIOR_REGRESSOR_COL.value
    residuals_scale_upper : float
    flat_multiplier : bool
        Default set as True. If False, we will adjust knot scale with a multiplier based on regressor volume
        around each knot; When True, set all multiplier as 1
    ktrlite_optim_args : dict
        the optimizing config for the ktrlite model (to fit level/seasonality). Default to be dict().
    """

    _data_input_mapper = DataInputMapper
    # stan or pyro model name (e.g. name of `*.stan` file in package)
    _model_name = "ktr"
    _supported_estimator_types = [PyroEstimatorSVI]

    def __init__(
        self,
        # level
        level_knot_scale=0.1,
        level_segments=10,
        level_knot_distance=None,
        level_knot_dates=None,
        # seasonality
        seasonality=None,
        seasonality_fs_order=None,
        seasonality_segments=2,
        seasonal_initial_knot_scale=1.0,
        seasonal_knot_scale=0.1,
        # regression
        regressor_col=None,
        regressor_sign=None,
        regressor_init_knot_loc=None,
        regressor_init_knot_scale=None,
        regressor_knot_scale=None,
        regression_segments=5,
        regression_knot_distance=None,
        regression_knot_dates=None,
        regression_rho=0.15,
        # shared
        degree_of_freedom=30,
        date_freq=None,
        # time-based coefficient priors
        coef_prior_list=None,
        flat_multiplier=True,
        residuals_scale_upper=None,
        ktrlite_optim_args=dict(),
        **kwargs,
    ):
        super().__init__(**kwargs)  # create estimator in base class

        # level configurations
        self.level_knot_scale = level_knot_scale
        self.level_segments = level_segments
        self.level_knot_distance = level_knot_distance
        self.level_knot_dates = level_knot_dates
        self._level_knot_dates = self.level_knot_dates
        self.level_knots = None
        self._level_knots = None

        self._kernel_level = None
        self._num_knots_level = None
        self.knots_tp_level = None

        # seasonality configurations
        self.seasonality = seasonality
        self.seasonality_fs_order = seasonality_fs_order
        self._seasonality = self.seasonality
        # used to name different seasonal components in prediction
        self._seasonality_fs_order = self.seasonality_fs_order
        self.seasonal_initial_knot_scale = seasonal_initial_knot_scale
        self.seasonal_knot_scale = seasonal_knot_scale
        self.seasonality_segments = seasonality_segments
        self._seas_term = 0

        self._seasonality_coef_knot_dates = None
        self._seasonality_coef_knots = None

        # regression configurations
        self.regressor_col = regressor_col
        self.regressor_sign = regressor_sign
        self.regressor_init_knot_loc = regressor_init_knot_loc
        self.regressor_init_knot_scale = regressor_init_knot_scale
        self.regressor_knot_scale = regressor_knot_scale
        self.regression_knot_distance = regression_knot_distance
        self.regression_segments = regression_segments
        self._regression_knot_dates = regression_knot_dates

        self.regression_rho = regression_rho
        self.flat_multiplier = flat_multiplier

        # set private var to arg value
        # if None set default in _set_default_args()
        self._regressor_sign = self.regressor_sign
        self._regressor_init_knot_loc = self.regressor_init_knot_loc
        self._regressor_init_knot_scale = self.regressor_init_knot_scale
        self._regressor_knot_scale = self.regressor_knot_scale

        self.coef_prior_list = coef_prior_list
        self._coef_prior_list = []
        self._regression_knots_idx = None
        self._num_of_regressors = 0

        # positive regressors
        self._num_of_positive_regressors = 0
        self._positive_regressor_col = list()
        self._positive_regressor_init_knot_loc = list()
        self._positive_regressor_init_knot_scale = list()
        self._positive_regressor_knot_scale_1d = list()
        self._positive_regressor_knot_scale = list()
        # negative regressors
        self._num_of_negative_regressors = 0
        self._negative_regressor_col = list()
        self._negative_regressor_init_knot_loc = list()
        self._negative_regressor_init_knot_scale = list()
        self._negative_regressor_knot_scale_1d = list()
        self._negative_regressor_knot_scale = list()
        # regular regressors
        self._num_of_regular_regressors = 0
        self._regular_regressor_col = list()
        self._regular_regressor_init_knot_loc = list()
        self._regular_regressor_init_knot_scale = list()
        self._regular_regressor_knot_scale_1d = list()
        self._regular_regressor_knot_scale = list()
        self._regressor_col = list()

        # init dynamic data attributes
        # the following are set by `_set_dynamic_attributes()` and generally set during fit()
        # from input df
        # response data
        self._is_valid_response = None
        self._which_valid_response = None
        self._num_of_valid_response = 0

        # regression data
        self._knots_tp_coefficients = None
        self._positive_regressor_matrix = None
        self._negative_regressor_matrix = None
        self._regular_regressor_matrix = None

        # other configurations
        self.date_freq = date_freq
        self.degree_of_freedom = degree_of_freedom
        self.residuals_scale_upper = residuals_scale_upper
        self._residuals_scale_upper = residuals_scale_upper
        self.ktrlite_optim_args = ktrlite_optim_args

        self._set_static_attributes()
        self._set_model_param_names()

    def _set_model_param_names(self):
        """Overriding base template functions. Model parameters to extract"""
        self._model_param_names += [param.value for param in BaseSamplingParameters]
        if self._num_of_regressors > 0:
            self._model_param_names += [
                param.value for param in RegressionSamplingParameters
            ]

    def _set_default_args(self):
        """Set default attributes for None"""
        # default checks for seasonality and seasonality_fs_order will be conducted
        # in ktrlite model and we will extract them from ktrlite model directly later
        if self.coef_prior_list is not None:
            self._coef_prior_list = deepcopy(self.coef_prior_list)

        # if no regressors, end here #
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
                    raise IllegalArgument(
                        "Wrong dimension length in Regression Param Input"
                    )

        # regressor defaults
        num_of_regressors = len(self.regressor_col)

        _validate_params_len(
            [
                self.regressor_sign,
                self.regressor_init_knot_loc,
                self.regressor_init_knot_scale,
                self.regressor_knot_scale,
            ],
            num_of_regressors,
        )

        if self.regressor_sign is None:
            self._regressor_sign = [DEFAULT_REGRESSOR_SIGN] * num_of_regressors

        if self.regressor_init_knot_loc is None:
            self._regressor_init_knot_loc = [
                DEFAULT_COEFFICIENTS_INIT_KNOT_LOC
            ] * num_of_regressors

        if self.regressor_init_knot_scale is None:
            self._regressor_init_knot_scale = [
                DEFAULT_COEFFICIENTS_INIT_KNOT_SCALE
            ] * num_of_regressors

        if self.regressor_knot_scale is None:
            self._regressor_knot_scale = [
                DEFAULT_COEFFICIENTS_KNOT_SCALE
            ] * num_of_regressors

        self._num_of_regressors = num_of_regressors

    def _set_static_regression_attributes(self):
        # if no regressors, end here
        if self._num_of_regressors == 0:
            return

        for index, reg_sign in enumerate(self._regressor_sign):
            if reg_sign == "+":
                self._num_of_positive_regressors += 1
                self._positive_regressor_col.append(self.regressor_col[index])
                # used for 'pr_knot_loc' sampling in pyro
                self._positive_regressor_init_knot_loc.append(
                    self._regressor_init_knot_loc[index]
                )
                self._positive_regressor_init_knot_scale.append(
                    self._regressor_init_knot_scale[index]
                )
                # used for 'pr_knot' sampling in pyro
                self._positive_regressor_knot_scale_1d.append(
                    self._regressor_knot_scale[index]
                )
            elif reg_sign == "-":
                self._num_of_negative_regressors += 1
                self._negative_regressor_col.append(self.regressor_col[index])
                # used for 'nr_knot_loc' sampling in pyro
                self._negative_regressor_init_knot_loc.append(
                    self._regressor_init_knot_loc[index]
                )
                self._negative_regressor_init_knot_scale.append(
                    self._regressor_init_knot_scale[index]
                )
                # used for 'nr_knot' sampling in pyro
                self._negative_regressor_knot_scale_1d.append(
                    self._regressor_knot_scale[index]
                )
            else:
                self._num_of_regular_regressors += 1
                self._regular_regressor_col.append(self.regressor_col[index])
                # used for 'rr_knot_loc' sampling in pyro
                self._regular_regressor_init_knot_loc.append(
                    self._regressor_init_knot_loc[index]
                )
                self._regular_regressor_init_knot_scale.append(
                    self._regressor_init_knot_scale[index]
                )
                # used for 'rr_knot' sampling in pyro
                self._regular_regressor_knot_scale_1d.append(
                    self._regressor_knot_scale[index]
                )
        # regular first, then positive, then negative
        self._regressor_col = (
            self._regular_regressor_col
            + self._positive_regressor_col
            + self._negative_regressor_col
        )
        # numpy conversion
        self._positive_regressor_init_knot_loc = np.array(
            self._positive_regressor_init_knot_loc
        )
        self._positive_regressor_init_knot_scale = np.array(
            self._positive_regressor_init_knot_scale
        )
        self._positive_regressor_knot_scale_1d = np.array(
            self._positive_regressor_knot_scale_1d
        )
        self._negative_regressor_init_knot_loc = np.array(
            self._negative_regressor_init_knot_loc
        )
        self._negative_regressor_init_knot_scale = np.array(
            self._negative_regressor_init_knot_scale
        )
        self._negative_regressor_knot_scale_1d = np.array(
            self._negative_regressor_knot_scale_1d
        )
        self._regular_regressor_init_knot_loc = np.array(
            self._regular_regressor_init_knot_loc
        )
        self._regular_regressor_init_knot_scale = np.array(
            self._regular_regressor_init_knot_scale
        )
        self._regular_regressor_knot_scale_1d = np.array(
            self._regular_regressor_knot_scale_1d
        )

    @staticmethod
    def _validate_coef_prior(coef_prior_list):
        for test_dict in coef_prior_list:
            if set(test_dict.keys()) != set(
                [
                    KTRTimePointPriorKeys.NAME.value,
                    KTRTimePointPriorKeys.PRIOR_START_TP_IDX.value,
                    KTRTimePointPriorKeys.PRIOR_END_TP_IDX.value,
                    KTRTimePointPriorKeys.PRIOR_MEAN.value,
                    KTRTimePointPriorKeys.PRIOR_SD.value,
                    KTRTimePointPriorKeys.PRIOR_REGRESSOR_COL.value,
                ]
            ):
                raise IllegalArgument("wrong key name in inserted prior dict")
            len_insert_prior = list()
            for key, val in test_dict.items():
                if key in [
                    KTRTimePointPriorKeys.PRIOR_MEAN.value,
                    KTRTimePointPriorKeys.PRIOR_SD.value,
                    KTRTimePointPriorKeys.PRIOR_REGRESSOR_COL.value,
                ]:
                    len_insert_prior.append(len(val))
            if not all(
                len_insert == len_insert_prior[0] for len_insert in len_insert_prior
            ):
                raise IllegalArgument("wrong dimension length in inserted prior dict")

    # @staticmethod
    # def _validate_level_knot_inputs(level_knot_dates, level_knots):
    #     if len(level_knots) != len(level_knot_dates):
    #         raise IllegalArgument('level_knots and level_knot_dates should have the same length')

    def _set_coef_prior_idx(self):
        if self._coef_prior_list and len(self._regressor_col) > 0:
            for x in self._coef_prior_list:
                prior_regressor_col_idx = [
                    np.where(np.array(self._regressor_col) == col)[0][0]
                    for col in x[KTRTimePointPriorKeys.PRIOR_REGRESSOR_COL.value]
                ]
                x.update({"prior_regressor_col_idx": prior_regressor_col_idx})

    def _set_static_attributes(self):
        """model data input based on args at instantiation or computed from args at instantiation"""
        self._set_default_args()
        self._set_static_regression_attributes()

        # self._validate_level_knot_inputs(self.level_knot_dates, self.level_knots)

        if self._coef_prior_list:
            self._validate_coef_prior(self._coef_prior_list)
            self._set_coef_prior_idx()

    def _set_valid_response_attributes(self, training_meta):
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        response = training_meta[TrainingMetaKeys.RESPONSE.value]

        if self._seasonality:
            max_seasonality = np.round(np.max(self._seasonality)).astype(int)
            if num_of_observations < max_seasonality:
                raise ModelException(
                    "Number of observations {} is less than max seasonality {}".format(
                        num_of_observations, max_seasonality
                    )
                )
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
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        # validate regression columns
        if self.regressor_col is not None and not set(self.regressor_col).issubset(
            df.columns
        ):
            raise ModelException(
                "DataFrame does not contain specified regressor column(s)."
            )

        # init of regression matrix depends on length of response vector
        self._positive_regressor_matrix = np.zeros(
            (num_of_observations, 0), dtype=np.double
        )
        self._negative_regressor_matrix = np.zeros(
            (num_of_observations, 0), dtype=np.double
        )
        self._regular_regressor_matrix = np.zeros(
            (num_of_observations, 0), dtype=np.double
        )

        # update regression matrices
        if self._num_of_positive_regressors > 0:
            self._positive_regressor_matrix = df.filter(
                items=self._positive_regressor_col,
            ).values

        if self._num_of_negative_regressors > 0:
            self._negative_regressor_matrix = df.filter(
                items=self._negative_regressor_col,
            ).values

        if self._num_of_regular_regressors > 0:
            self._regular_regressor_matrix = df.filter(
                items=self._regular_regressor_col,
            ).values

    def _set_coefficients_kernel_matrix(self, df, training_meta):
        """Derive knots position and kernel matrix and other related meta data"""
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        # date_col = training_meta[TrainingMetaKeys.DATE_COL.value]

        # placeholder
        self._kernel_coefficients = np.zeros((num_of_observations, 0), dtype=np.double)
        self._num_knots_coefficients = 0

        if self._num_of_regressors > 0:
            self._regression_knots_idx = get_knot_idx(
                date_array=date_array,
                num_of_obs=num_of_observations,
                knot_dates=self._regression_knot_dates,
                knot_distance=self.regression_knot_distance,
                num_of_segments=self.regression_segments,
            )

            tp = np.arange(1, num_of_observations + 1) / num_of_observations
            self._knots_tp_coefficients = (
                1 + self._regression_knots_idx
            ) / num_of_observations
            self._kernel_coefficients = gauss_kernel(
                tp, self._knots_tp_coefficients, rho=self.regression_rho
            )
            self._num_knots_coefficients = len(self._knots_tp_coefficients)
            if self.date_freq is None:
                self.date_freq = pd.infer_freq(date_array)
                # self.date_timedelta = date_array.diff().min()
            self._regression_knot_dates = get_knot_dates(
                date_array[0], self._regression_knots_idx, self.date_freq
            )

    def _set_knots_scale_matrix(self, df, training_meta):
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        if self._num_of_positive_regressors > 0:
            # calculate average local absolute volume for each segment
            local_val = np.ones(
                (self._num_of_positive_regressors, self._num_knots_coefficients)
            )
            if self.flat_multiplier:
                multiplier = np.ones(local_val.shape)
            else:
                multiplier = np.ones(local_val.shape)
                # store local value for the range on the left side since last knot
                for idx in range(len(self._regression_knots_idx)):
                    if idx < len(self._regression_knots_idx) - 1:
                        str_idx = self._regression_knots_idx[idx]
                        end_idx = self._regression_knots_idx[idx + 1]
                    else:
                        str_idx = self._regression_knots_idx[idx]
                        end_idx = num_of_observations

                    local_val[:, idx] = np.mean(
                        np.fabs(self._positive_regressor_matrix[str_idx:end_idx]),
                        axis=0,
                    )

                global_mean = np.expand_dims(
                    np.mean(np.fabs(self._positive_regressor_matrix), axis=0), -1
                )
                test_flag = local_val < 0.01 * global_mean

                # adjust knot scale with the multiplier derive by the average value and shift by 0.001 to avoid zeros in
                # scale parameters
                multiplier[test_flag] = DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER
                # replace entire row of nan (when 0.1 * global_mean is equal to global_min) with upper bound
                multiplier[np.isnan(multiplier).all(axis=-1)] = 1.0

            # geometric drift i.e. 0.1 = 10% up-down in 1 s.d. prob.
            # self._positive_regressor_knot_scale has shape num_of_pr x num_of_knot
            self._positive_regressor_knot_scale = multiplier * np.expand_dims(
                self._positive_regressor_knot_scale_1d, -1
            )
            # keep a lower bound of scale parameters
            self._positive_regressor_knot_scale[
                self._positive_regressor_knot_scale < 1e-4
            ] = 1e-4
            # TODO: we change the type here, maybe we should change it earlier?
            self._positive_regressor_init_knot_scale = np.array(
                self._positive_regressor_init_knot_scale
            )
            self._positive_regressor_init_knot_scale[
                self._positive_regressor_init_knot_scale < 1e-4
            ] = 1e-4

        if self._num_of_negative_regressors > 0:
            # calculate average local absolute volume for each segment
            local_val = np.ones(
                (self._num_of_negative_regressors, self._num_knots_coefficients)
            )
            if self.flat_multiplier:
                multiplier = np.ones(local_val.shape)
            else:
                multiplier = np.ones(local_val.shape)
                # store local value for the range on the left side since last knot
                for idx in range(len(self._regression_knots_idx)):
                    if idx < len(self._regression_knots_idx) - 1:
                        str_idx = self._regression_knots_idx[idx]
                        end_idx = self._regression_knots_idx[idx + 1]
                    else:
                        str_idx = self._regression_knots_idx[idx]
                        end_idx = num_of_observations

                    local_val[:, idx] = np.mean(
                        np.fabs(self._negative_regressor_matrix[str_idx:end_idx]),
                        axis=0,
                    )

                global_mean = np.expand_dims(
                    np.mean(np.fabs(self._negative_regressor_matrix), axis=0), -1
                )
                test_flag = local_val < 0.01 * global_mean

                # adjust knot scale with the multiplier derive by the average value and shift by 0.001 to avoid zeros in
                # scale parameters
                multiplier[test_flag] = DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER
                # replace entire row of nan (when 0.1 * global_mean is equal to global_min) with upper bound
                multiplier[np.isnan(multiplier).all(axis=-1)] = 1.0

            # geometric drift i.e. 0.1 = 10% up-down in 1 s.d. prob.
            self._negative_regressor_knot_scale = multiplier * np.expand_dims(
                self._negative_regressor_knot_scale_1d, -1
            )
            # keep a lower bound of scale parameters
            self._negative_regressor_knot_scale[
                self._negative_regressor_knot_scale < 1e-4
            ] = 1e-4
            # TODO: we change the type here, maybe we should change it earlier?
            self._negative_regressor_init_knot_scale = np.array(
                self._negative_regressor_init_knot_scale
            )
            self._negative_regressor_init_knot_scale[
                self._negative_regressor_init_knot_scale < 1e-4
            ] = 1e-4

        if self._num_of_regular_regressors > 0:
            # do the same for regular regressor
            # calculate average local absolute volume for each segment
            local_val = np.ones(
                (self._num_of_regular_regressors, self._num_knots_coefficients)
            )
            if self.flat_multiplier:
                multiplier = np.ones(local_val.shape)
            else:
                multiplier = np.ones(local_val.shape)
                # store local value for the range on the left side since last knot
                for idx in range(len(self._regression_knots_idx)):
                    if idx < len(self._regression_knots_idx) - 1:
                        str_idx = self._regression_knots_idx[idx]
                        end_idx = self._regression_knots_idx[idx + 1]
                    else:
                        str_idx = self._regression_knots_idx[idx]
                        end_idx = num_of_observations

                    local_val[:, idx] = np.mean(
                        np.fabs(self._regular_regressor_matrix[str_idx:end_idx]), axis=0
                    )

            # adjust knot scale with the multiplier derive by the average value and shift by 0.001 to avoid zeros in
            # scale parameters
            global_mean = np.expand_dims(
                np.mean(np.fabs(self._regular_regressor_matrix), axis=0), -1
            )
            test_flag = local_val < 0.01 * global_mean
            multiplier[test_flag] = DEFAULT_LOWER_BOUND_SCALE_MULTIPLIER
            # replace entire row of nan (when 0.1 * global_mean is equal to global_min) with upper bound
            multiplier[np.isnan(multiplier).all(axis=-1)] = 1.0

            # geometric drift i.e. 0.1 = 10% up-down in 1 s.d. prob.
            # self._regular_regressor_knot_scale has shape num_of_pr x num_of_knot
            self._regular_regressor_knot_scale = multiplier * np.expand_dims(
                self._regular_regressor_knot_scale_1d, -1
            )
            # keep a lower bound of scale parameters
            self._regular_regressor_knot_scale[
                self._regular_regressor_knot_scale < 1e-4
            ] = 1e-4
            # TODO: we change the type here, maybe we should change it earlier?
            self._regular_regressor_init_knot_scale = np.array(
                self._regular_regressor_init_knot_scale
            )
            self._regular_regressor_init_knot_scale[
                self._regular_regressor_init_knot_scale < 1e-4
            ] = 1e-4

    def _generate_tp(self, training_meta, prediction_date_array):
        """Used in _generate_seas"""
        training_end = training_meta[TrainingMetaKeys.END.value]
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        prediction_start = prediction_date_array[0]
        output_len = len(prediction_date_array)
        if prediction_start > training_end:
            start = num_of_observations
        else:
            start = pd.Index(date_array).get_loc(prediction_start)

        new_tp = np.arange(start + 1, start + output_len + 1) / num_of_observations
        return new_tp

    def _generate_insample_tp(self, training_meta, date_array):
        """Used in _generate_seas"""
        train_date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        idx = np.nonzero(np.in1d(train_date_array, date_array))[0]
        tp = (idx + 1) / num_of_observations
        return tp

    # def _generate_coefs(self, training_meta, prediction_date_array, coef_knot_dates, coef_knot):
    #     """Used in _generate_seas"""
    #     new_tp = self._generate_tp(training_meta, prediction_date_array)
    #     knots_tp_coef = self._generate_insample_tp(training_meta, coef_knot_dates)
    #     kernel_coef = sandwich_kernel(new_tp, knots_tp_coef)
    #     coefs = np.squeeze(np.matmul(coef_knot, kernel_coef.transpose(1, 0)), axis=0).transpose(1, 0)
    #     return coefs

    def _generate_seas(
        self,
        df,
        training_meta,
        coef_knot_dates,
        coef_knots,
        seasonality,
        seasonality_fs_order,
        seasonality_labels,
    ):
        """To calculate the seasonality term based on the _seasonal_knots_input.
        Parameters
        ----------
        df : pd.DataFrame
            input df
        training_meta: dict
            meta dictionary for the training input
        coef_knot_dates : 1-D array like
            dates for seasonality coefficient knots
        coef_knots : dict
            dict of seasonal coefficient knots from each seasonality
        seasonality : list
            seasonality input; list of float
        seasonality_fs_order : list
            seasonality_fs_order input list of int

        Returns
        -----------
        dict :
            a dictionary contains seasonal regression components mapped by each seasonality
        """
        df = df.copy()
        # store each component as a dictionary
        seas_decomp = dict()

        if seasonality is not None and len(seasonality) > 0:
            date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
            date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
            training_end = training_meta[TrainingMetaKeys.END.value]
            num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]

            prediction_date_array = df[date_col].values
            prediction_start = prediction_date_array[0]

            if prediction_start > training_end:
                # time index for prediction start
                start = num_of_observations
            else:
                # time index for prediction start
                start = pd.Index(date_array).get_loc(prediction_start)

            # dictionary
            seas_regressors = make_seasonal_regressors(
                n=df.shape[0],
                periods=seasonality,
                orders=seasonality_fs_order,
                labels=seasonality_labels,
                shift=start,
            )

            new_tp = self._generate_tp(training_meta, prediction_date_array)
            knots_tp_coef = self._generate_insample_tp(training_meta, coef_knot_dates)
            coef_kernel = sandwich_kernel(new_tp, knots_tp_coef)

            # init of regression matrix depends on length of response vector
            total_seas_regression = np.zeros((1, df.shape[0]), dtype=np.double)

            for k in seasonality_labels:
                seas_regresor_matrix = seas_regressors[k]
                coef_knot = coef_knots[k]
                # time-step x coefficients
                seas_coef = np.squeeze(
                    np.matmul(coef_knot, coef_kernel.transpose(1, 0)), axis=0
                ).transpose(1, 0)
                seas_regression = np.sum(seas_coef * seas_regresor_matrix, axis=-1)
                seas_decomp[k] = np.expand_dims(seas_regression, 0)
                total_seas_regression += seas_regression
        else:
            total_seas_regression = np.zeros((1, df.shape[0]), dtype=np.double)

        return total_seas_regression, seas_decomp

    def _set_levs_and_seas(self, df, training_meta):
        response_col = training_meta["response_col"]
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]

        # use ktrlite to derive levs and seas
        ktrlite = KTRLite(
            response_col=response_col,
            date_col=date_col,
            level_knot_scale=self.level_knot_scale,
            level_segments=self.level_segments,
            level_knot_dates=self.level_knot_dates,
            level_knot_distance=self.level_knot_distance,
            seasonality=self.seasonality,
            seasonality_fs_order=self.seasonality_fs_order,
            seasonal_initial_knot_scale=self.seasonal_initial_knot_scale,
            seasonal_knot_scale=self.seasonal_knot_scale,
            seasonality_segments=self.seasonality_segments,
            degree_of_freedom=self.degree_of_freedom,
            date_freq=self.date_freq,
            estimator="stan-map",
            **self.ktrlite_optim_args,
        )
        ktrlite.fit(df=df)
        # self._ktrlite_model = ktrlite
        ktrlite_pt_posteriors = ktrlite.get_point_posteriors()
        ktrlite_obs_scale = ktrlite_pt_posteriors["map"]["obs_scale"]

        # load _seasonality and _seasonality_fs_order
        self._seasonality = ktrlite._model._seasonality
        self._seasonality_fs_order = ktrlite._model._seasonality_fs_order
        self._seasonality_labels = list()
        for seas in self._seasonality:
            self._seasonality_labels.append("seasonality_{}".format(seas))

        # if input None for upper bound of residuals scale, use data-driven input
        if self.residuals_scale_upper is None:
            # make it 5 times to have some buffer in case we over-fit in KTRLite
            self._residuals_scale_upper = min(
                ktrlite_obs_scale * 5, training_meta["response_sd"]
            )

        # this part is to extract level and seasonality result from KTRLite
        self._level_knots = np.squeeze(ktrlite_pt_posteriors["map"]["lev_knot"])
        self._level_knot_dates = ktrlite._model._level_knot_dates
        tp = np.arange(1, num_of_observations + 1) / num_of_observations
        # # trim level knots dates when they are beyond training dates
        # lev_knot_dates = list()
        # lev_knots = list()
        # for i, x in enumerate(self.level_knot_dates):
        #     if (x <= df[date_col].max()) and (x >= df[date_col].min()):
        #         lev_knot_dates.append(x)
        #         lev_knots.append(self._level_knots[i])
        # self._level_knot_dates = pd.to_datetime(lev_knot_dates)
        # self._level_knots = np.array(lev_knots)

        self._level_knots_idx = get_knot_idx(
            date_array=date_array,
            num_of_obs=None,
            knot_dates=self._level_knot_dates,
            knot_distance=None,
            num_of_segments=None,
        )
        self.knots_tp_level = (1 + self._level_knots_idx) / num_of_observations
        self._kernel_level = sandwich_kernel(tp, self.knots_tp_level)
        self._num_knots_level = len(self._level_knot_dates)

        if self._seasonality:
            self._seasonality_coef_knot_dates = ktrlite._model._coef_knot_dates
            coef_knots_flatten = ktrlite_pt_posteriors["map"]["coef_knot"]
            coef_knots = dict()
            pos = 0
            for idx, label in enumerate(self._seasonality_labels):
                order = self._seasonality_fs_order[idx]
                coef_knots[label] = coef_knots_flatten[..., pos : (pos + 2 * order), :]
                pos += 2 * order
            self._seasonality_coef_knots = coef_knots

            # we just need total here and because of
            self._seas_term, _ = self._generate_seas(
                df,
                training_meta,
                self._seasonality_coef_knot_dates,
                self._seasonality_coef_knots,
                self._seasonality,
                self._seasonality_fs_order,
                self._seasonality_labels,
            )
            # remove batch size as an input for models
            self._seas_term = np.squeeze(self._seas_term, 0)

    def _filter_coef_prior(self, df):
        if self._coef_prior_list and len(self._regressor_col) > 0:
            # iterate over a copy due to the removal operation
            for test_dict in self._coef_prior_list[:]:
                prior_regressor_col = test_dict[
                    KTRTimePointPriorKeys.PRIOR_REGRESSOR_COL.value
                ]
                m = test_dict[KTRTimePointPriorKeys.PRIOR_MEAN.value]
                sd = test_dict[KTRTimePointPriorKeys.PRIOR_SD.value]
                end_tp_idx = min(
                    test_dict[KTRTimePointPriorKeys.PRIOR_END_TP_IDX.value], df.shape[0]
                )
                start_tp_idx = min(
                    test_dict[KTRTimePointPriorKeys.PRIOR_START_TP_IDX.value],
                    df.shape[0],
                )
                if start_tp_idx < end_tp_idx:
                    expected_shape = (
                        end_tp_idx - start_tp_idx,
                        len(prior_regressor_col),
                    )
                    test_dict.update(
                        {KTRTimePointPriorKeys.PRIOR_END_TP_IDX.value: end_tp_idx}
                    )
                    test_dict.update(
                        {KTRTimePointPriorKeys.PRIOR_START_TP_IDX.value: start_tp_idx}
                    )
                    # mean/sd expanding
                    test_dict.update(
                        {
                            KTRTimePointPriorKeys.PRIOR_MEAN.value: np.full(
                                expected_shape, m
                            )
                        }
                    )
                    test_dict.update(
                        {
                            KTRTimePointPriorKeys.PRIOR_SD.value: np.full(
                                expected_shape, sd
                            )
                        }
                    )
                else:
                    # removing invalid prior
                    self._coef_prior_list.remove(test_dict)

    def set_dynamic_attributes(self, df, training_meta):
        """Overriding: func: `~orbit.models.BaseETS._set_dynamic_attributes"""
        self._set_regressor_matrix(df, training_meta)
        self._set_coefficients_kernel_matrix(df, training_meta)
        self._set_knots_scale_matrix(df, training_meta)
        self._set_levs_and_seas(df, training_meta)
        self._filter_coef_prior(df)
        self._set_valid_response_attributes(training_meta)

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

    def predict(
        self,
        posterior_estimates,
        df,
        training_meta,
        prediction_meta,
        coefficient_method="smooth",
        include_error=False,
        store_prediction_array=False,
        **kwargs,
    ):
        """Vectorized version of prediction math
        Parameters
        ----
        coefficient_method : str
            either "smooth" or "empirical". when "empirical" is used, curves are sampled/aggregated directly
            from beta posteriors; when "smooth" is used, first extract sampled/aggregated posteriors of knots
            then beta.
            this mainly impacts the aggregated estimation method; full bayesian should not be impacted
        include_error : bool
            if generating the noise samples
        store_prediction_array : bool
            if storing the prediction array
        """
        ################################################################
        # Model Attributes
        ################################################################
        # FIXME: do we still need this?
        model = deepcopy(posterior_estimates)
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        ################################################################
        # Prediction Attributes
        ################################################################
        output_len = prediction_meta[PredictionMetaKeys.PREDICTION_DF_LEN.value]
        prediction_start = prediction_meta[PredictionMetaKeys.START.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        training_end = training_meta[TrainingMetaKeys.END.value]

        # Here assume dates are ordered and consecutive
        # if prediction_meta[PredictionMetaKeys.START.value] > self.training_end,
        # assume prediction starts right after train end
        if prediction_start > training_end:
            # time index for prediction start
            start = num_of_observations
        else:
            start = pd.Index(date_array).get_loc(prediction_start)

        new_tp = np.arange(start + 1, start + output_len + 1) / num_of_observations
        if include_error:
            # in-sample knots
            lev_knot_in = model.get(BaseSamplingParameters.LEVEL_KNOT.value)
            # TODO: hacky way; let's just assume last two knot distance is knots distance for all knots
            lev_knot_width = self.knots_tp_level[-1] - self.knots_tp_level[-2]
            # check whether we need to put new knots for simulation
            if new_tp[-1] >= self.knots_tp_level[-1] + lev_knot_width:
                # derive knots tp
                knots_tp_level_out = np.arange(
                    self.knots_tp_level[-1] + lev_knot_width, new_tp[-1], lev_knot_width
                )
                new_knots_tp_level = np.concatenate(
                    [self.knots_tp_level, knots_tp_level_out]
                )
                lev_knot_out = np.random.laplace(
                    0,
                    self.level_knot_scale,
                    size=(lev_knot_in.shape[0], len(knots_tp_level_out)),
                )
                lev_knot_out = np.cumsum(
                    np.concatenate(
                        [lev_knot_in[:, -1].reshape(-1, 1), lev_knot_out], axis=1
                    ),
                    axis=1,
                )[:, 1:]
                lev_knot = np.concatenate([lev_knot_in, lev_knot_out], axis=1)
            else:
                new_knots_tp_level = self.knots_tp_level
                lev_knot = lev_knot_in
            kernel_level = sandwich_kernel(new_tp, new_knots_tp_level)
        else:
            lev_knot = model.get(BaseSamplingParameters.LEVEL_KNOT.value)
            kernel_level = sandwich_kernel(new_tp, self.knots_tp_level)
        obs_scale = model.get(BaseSamplingParameters.OBS_SCALE.value)
        obs_scale = obs_scale.reshape(-1, 1)

        # if self._seasonality is not None:
        # condition of seasonality is checked inside
        total_seas, seas_decomp = self._generate_seas(
            df,
            training_meta,
            self._seasonality_coef_knot_dates,
            self._seasonality_coef_knots,
            self._seasonality,
            self._seasonality_fs_order,
            self._seasonality_labels,
        )

        #     # seas is 1-d array, add the batch size back
        #     seas = np.expand_dims(seas, 0)
        # else:
        #     # follow component shapes
        #     seas = np.zeros((1, output_len))

        trend = np.matmul(lev_knot, kernel_level.transpose((1, 0)))
        regression = np.zeros(trend.shape)
        if self._num_of_regressors > 0:
            regressor_matrix = df.filter(
                items=self._regressor_col,
            ).values
            regressor_betas = self._get_regression_coefs_matrix(
                training_meta,
                posterior_estimates,
                coefficient_method,
                date_array=prediction_meta[TrainingMetaKeys.DATE_ARRAY.value],
            )
            regression = np.sum(regressor_betas * regressor_matrix, axis=-1)

        if include_error:
            epsilon = nct.rvs(
                self.degree_of_freedom,
                nc=0,
                loc=0,
                scale=obs_scale,
                size=(num_sample, len(new_tp)),
            )
            trend += epsilon

        pred_array = trend + total_seas + regression

        # if decompose output dictionary of components
        decomp_dict = {
            "prediction": pred_array,
            "trend": trend,
            "regression": regression,
        }

        # this is an input from ktrlite
        decomp_dict.update(seas_decomp)

        if store_prediction_array:
            self.pred_array = pred_array
        else:
            self.pred_array = None

        return decomp_dict

    def _get_regression_coefs_matrix(
        self, training_meta, posteriors, coefficient_method="smooth", date_array=None
    ):
        """internal function to provide coefficient matrix given a date array
        Args
        ----
        posteriors : dict
            posterior samples
        date_array : array like
            array of date stamp
        coefficient_method : str
            either "smooth" or "empirical". when "empirical" is used, curves are sampled/aggregated directly
            from beta posteriors; when "smooth" is used, first extract sampled/aggregated posteriors of knots
            then beta.
            this mainly impacts the aggregated estimation method; full bayesian should not be impacted.
        """
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        training_start = training_meta[TrainingMetaKeys.START.value]
        training_end = training_meta[TrainingMetaKeys.END.value]
        train_date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]

        if (
            self._num_of_regular_regressors
            + self._num_of_positive_regressors
            + self._num_of_negative_regressors
            == 0
        ):
            return None

        # if date_array not specified, coefficients in the training period will be retrieved
        if date_array is None:
            if coefficient_method == "smooth":
                coef_knots = posteriors.get(
                    RegressionSamplingParameters.COEFFICIENTS_KNOT.value
                )
                if len(self._regressor_col) == 1:
                    coef_knots = np.expand_dims(coef_knots, 1)
                # only 1 knot for 0 segments
                if self.regression_segments == 0:
                    coef_knots = np.expand_dims(coef_knots, -1)

                # result in batch x time step x regressor size shape
                regressor_betas = np.matmul(
                    coef_knots, self._kernel_coefficients.transpose((1, 0))
                )

                # if len(self._regressor_col) == 1:
                #     regressor_betas = np.expand_dims(regressor_betas, 0)
                regressor_betas = regressor_betas.transpose((0, 2, 1))
            elif coefficient_method == "empirical":
                regressor_betas = posteriors.get(
                    RegressionSamplingParameters.COEFFICIENTS.value
                )
            else:
                raise IllegalArgument(
                    "Wrong coefficient_method:{}".format(coefficient_method)
                )
        else:
            date_array = pd.to_datetime(date_array).values
            output_len = len(date_array)
            train_len = num_of_observations
            # some validation of date array
            if not is_ordered_datetime(date_array):
                raise IllegalArgument("Datetime index must be ordered and not repeat")
            prediction_start = date_array[0]

            if prediction_start < training_start:
                raise PredictionException(
                    "Prediction start must be after training start."
                )

            # If we cannot find a match of prediction range, assume prediction starts right after train end
            if prediction_start > training_end:
                # time index for prediction start
                start = train_len
                coef_repeats = [0] * (start - 1) + [output_len]
            else:
                # time index for prediction start
                start = pd.Index(train_date_array).get_loc(prediction_start)
                if output_len <= train_len - start:
                    coef_repeats = (
                        [0] * start
                        + [1] * output_len
                        + [0] * (train_len - start - output_len)
                    )
                else:
                    coef_repeats = (
                        [0] * start
                        + [1] * (train_len - start - 1)
                        + [output_len - train_len + start + 1]
                    )
            new_tp = np.arange(start + 1, start + output_len + 1) / num_of_observations

            if coefficient_method == "smooth":
                kernel_coefficients = gauss_kernel(
                    new_tp, self._knots_tp_coefficients, rho=self.regression_rho
                )

                coef_knots = posteriors.get(
                    RegressionSamplingParameters.COEFFICIENTS_KNOT.value
                )
                if len(self._regressor_col) == 1:
                    coef_knots = np.expand_dims(coef_knots, 1)
                # only 1 knot for 0 segments
                if self.regression_segments == 0:
                    coef_knots = np.expand_dims(coef_knots, -1)

                regressor_betas = np.matmul(
                    coef_knots, kernel_coefficients.transpose((1, 0))
                )
                if len(regressor_betas.shape) == 2:
                    regressor_betas = np.expand_dims(regressor_betas, 0)
                regressor_betas = regressor_betas.transpose((0, 2, 1))
            elif coefficient_method == "empirical":
                regressor_betas = posteriors.get(
                    RegressionSamplingParameters.COEFFICIENTS.value
                )
                regressor_betas = np.repeat(
                    regressor_betas, repeats=coef_repeats, axis=1
                )
            else:
                raise IllegalArgument(
                    "Wrong coefficient_method:{}".format(coefficient_method)
                )

        return regressor_betas

    def get_regression_coefs(
        self,
        training_meta,
        point_method,
        point_posteriors,
        posterior_samples,
        coefficient_method="smooth",
        date_array=None,
        include_ci=False,
        lower=0.05,
        upper=0.95,
    ):
        """Return DataFrame regression coefficients.

        Parameters
        ----------
        coefficient_method : str
            either "smooth" or "empirical". when "empirical" is used, curves are sampled/aggregated directly
            from beta posteriors; when "smooth" is used, first extract sampled/aggregated posteriors of knots
            then beta.
        date_array : array-like
            the list of dates for which the regressio coefficients will be reported.
            Default to be None. When it's None, all the dates in the training data will be used.
        include_ci : bool
            if including the confidence intervals for the regression coefficients
        lower : float between (0, 1). default to be 0.05
            lower bound for the CI
        upper : float between (0, 1). default to be 0.95.
            upper bound for the CI

        Returns
        -------
        Pandas data frame holding the dynamic regression coefficients
        """
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        reg_df = pd.DataFrame()
        if self._num_of_regressors == 0:
            return reg_df

        _point_method = point_method
        if point_method is None:
            _point_method = PredictMethod.MEDIAN.value

        posteriors = point_posteriors.get(_point_method)
        coefs = np.squeeze(
            self._get_regression_coefs_matrix(
                training_meta,
                posteriors,
                coefficient_method=coefficient_method,
                date_array=date_array,
            )
        )
        if len(coefs.shape) == 1:
            coefs = np.expand_dims(coefs, -1)

        reg_df = pd.DataFrame(data=coefs, columns=self._regressor_col)
        if date_array is not None:
            reg_df[date_col] = date_array
        else:
            reg_df[date_col] = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        # re-arrange columns
        reg_df = reg_df[[date_col] + self._regressor_col]

        if include_ci:
            posteriors = posterior_samples
            coefs = self._get_regression_coefs_matrix(
                training_meta,
                posteriors,
                coefficient_method=coefficient_method,
                date_array=date_array,
            )
            coefficients_lower = np.quantile(coefs, lower, axis=0)
            coefficients_upper = np.quantile(coefs, upper, axis=0)
            reg_df_lower = reg_df.copy()
            reg_df_upper = reg_df.copy()
            for idx, col in enumerate(self._regressor_col):
                reg_df_lower[col] = coefficients_lower[:, idx]
                reg_df_upper[col] = coefficients_upper[:, idx]
            return reg_df, reg_df_lower, reg_df_upper

        return reg_df

    def get_regression_coef_knots(
        self, training_meta, point_method, point_posteriors, posterior_samples
    ):
        """Return DataFrame regression coefficient knots"""
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        _point_method = point_method
        if point_method is None:
            _point_method = PredictMethod.MEDIAN.value

        # init dataframe
        knots_df = pd.DataFrame()
        # end if no regressors
        if (
            self._num_of_regular_regressors
            + self._num_of_positive_regressors
            + self._num_of_negative_regressors
            == 0
        ):
            return knots_df

        knots_df[date_col] = self._regression_knot_dates
        # TODO: make the label as a constant
        knots_df["step"] = self._regression_knots_idx
        # batch size x regressor size x knot size
        coef_knots = point_posteriors.get(_point_method).get(
            RegressionSamplingParameters.COEFFICIENTS_KNOT.value
        )

        if len(self._regressor_col) == 1:
            coef_knots = np.expand_dims(coef_knots, 1)
        # only 1 knot for 0 segments
        if self.regression_segments == 0:
            coef_knots = np.expand_dims(coef_knots, -1)

        for idx, col in enumerate(self._regressor_col):
            knots_df[col] = np.transpose(coef_knots[:, idx])

        return knots_df

    @orbit_style_decorator
    def plot_regression_coefs(
        self,
        training_meta,
        point_method,
        point_posteriors,
        posterior_samples,
        coefficient_method="smooth",
        date_array=None,
        include_ci=False,
        lower=0.05,
        upper=0.95,
        with_knot=False,
        is_visible=True,
        ncol=2,
        ylim=None,
        markersize=200,
        figsize=(16, 8),
    ):
        """Plot regression coefficients.

        Parameters
        ----------
        coefficient_method : str
            either "smooth" or "empirical". when "empirical" is used, curves are sampled/aggregated directly
            from beta posteriors; when "smooth" is used, first extract sampled/aggregated posteriors of knots
            then beta.
        date_array : array-like
            the list of dates for which the regressio coefficients will be reported.
            Default to be None. When it's None, all the dates in the training data will be used.
        include_ci : bool
            if including the confidence intervals for the regression coefficients
        lower : float between (0, 1). default to be 0.05
            lower bound for the CI
        upper : float between (0, 1). default to be 0.95.
            upper bound for the CI
        with_knot : bool
            if plotting the regression knots in the graph
        ncol : int
            number of columns of the panel grid
                is_visible : boolean
            whether we want to show the plot. If called from unittest, is_visible might = False.
        is_visible : bool
            whether we want to show the plot. If called from unittest, is_visible might = False.
        markersize : int; optional
            knot marker size
        figsize : tuple; optional
            figsize passed to `matplotlib.pyplot.figure()`

        """
        # assume your first column is the date; this way can use a static method
        if include_ci:
            coef_df, coef_df_lower, coef_df_upper = self.get_regression_coefs(
                training_meta,
                point_method,
                point_posteriors,
                posterior_samples,
                coefficient_method=coefficient_method,
                date_array=date_array,
                include_ci=include_ci,
                lower=lower,
                upper=upper,
            )
        else:
            coef_df = self.get_regression_coefs(
                training_meta,
                point_method,
                point_posteriors,
                posterior_samples,
                coefficient_method=coefficient_method,
                date_array=date_array,
                include_ci=include_ci,
                lower=lower,
                upper=upper,
            )
            coef_df_lower, coef_df_upper = None, None
        if with_knot:
            knot_df = self.get_regression_coef_knots(
                training_meta, point_method, point_posteriors, posterior_samples
            )
        else:
            knot_df = None

        regressor_col = coef_df.columns.tolist()[1:]
        nrow = math.ceil(len(regressor_col) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

        for idx, col in enumerate(regressor_col):
            row_idx = idx // ncol
            col_idx = idx % ncol
            coef = coef_df[col]
            axes[row_idx, col_idx].plot(
                coef, alpha=0.8, label="coefficients", color=OrbitPalette.BLUE.value
            )
            if coef_df_lower is not None and coef_df_upper is not None:
                coef_lower = coef_df_lower[col]
                coef_upper = coef_df_upper[col]
                axes[row_idx, col_idx].fill_between(
                    np.arange(0, coef_df.shape[0]),
                    coef_lower,
                    coef_upper,
                    alpha=0.3,
                    color=OrbitPalette.BLUE.value,
                )
            if knot_df is not None:
                step = knot_df["step"]
                knots = knot_df[col].values
                axes[row_idx, col_idx].scatter(
                    x=step,
                    y=knots,
                    marker="^",
                    s=markersize,
                    color=OrbitPalette.GREEN.value,
                    alpha=0.5,
                )
            if ylim is not None:
                axes[row_idx, col_idx].set_ylim(ylim)
            axes[row_idx, col_idx].set_title("{}".format(col))
            axes[row_idx, col_idx].ticklabel_format(useOffset=False)

        plt.tight_layout()

        if is_visible:
            plt.show()
        else:
            plt.close()

        return axes

    # TODO: need a unit test of this function
    def get_level_knots(
        self, training_meta, point_method, point_posteriors, posterior_samples
    ):
        """Given posteriors, return knots and correspondent date"""
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        _point_method = point_method
        if point_method is None:
            _point_method = PredictMethod.MEDIAN.value
        lev_knots = point_posteriors.get(_point_method).get(
            BaseSamplingParameters.LEVEL_KNOT.value
        )
        lev_knots = np.squeeze(lev_knots, 0)
        out = {
            date_col: self._level_knot_dates,
            BaseSamplingParameters.LEVEL_KNOT.value: lev_knots,
        }

        return pd.DataFrame(out)

    def get_levels(
        self, training_meta, point_method, point_posteriors, posterior_samples
    ):
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        _point_method = point_method
        if point_method is None:
            _point_method = PredictMethod.MEDIAN.value
        levs = point_posteriors.get(_point_method).get(
            BaseSamplingParameters.LEVEL.value
        )
        levs = np.squeeze(levs, 0)
        out = {
            date_col: date_array,
            BaseSamplingParameters.LEVEL.value: levs,
        }

        return pd.DataFrame(out)

    @orbit_style_decorator
    def plot_lev_knots(
        self,
        training_meta,
        point_method,
        point_posteriors,
        posterior_samples,
        path=None,
        is_visible=True,
        title="",
        fontsize=16,
        markersize=250,
        figsize=(16, 8),
    ):
        """Plot the fitted level knots along with the actual time series.
         Parameters
         ----------
         path : str; optional
             path to save the figure
         is_visible : boolean
             whether we want to show the plot. If called from unittest, is_visible might = False.
         title : str; optional
             title of the plot
         fontsize : int; optional
             fontsize of the title
         markersize : int; optional
             knot marker size
         figsize : tuple; optional
             figsize passed to `matplotlib.pyplot.figure()`
        Returns
         -------
             matplotlib axes object
        """
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]
        response = training_meta[TrainingMetaKeys.RESPONSE.value]

        levels_df = self.get_levels(
            training_meta, point_method, point_posteriors, posterior_samples
        )
        knots_df = self.get_level_knots(
            training_meta, point_method, point_posteriors, posterior_samples
        )

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(
            date_array,
            response,
            color=OrbitPalette.BLUE.value,
            lw=1,
            alpha=0.7,
            label="actual",
        )
        ax.plot(
            levels_df[date_col],
            levels_df[BaseSamplingParameters.LEVEL.value],
            color=OrbitPalette.BLACK.value,
            lw=1,
            alpha=0.8,
            label=BaseSamplingParameters.LEVEL.value,
        )
        ax.scatter(
            knots_df[date_col],
            knots_df[BaseSamplingParameters.LEVEL_KNOT.value],
            color=OrbitPalette.GREEN.value,
            lw=1,
            s=markersize,
            marker="^",
            alpha=0.8,
            label=BaseSamplingParameters.LEVEL_KNOT.value,
        )
        ax.legend()
        ax.grid(True, which="major", c="grey", ls="-", lw=1, alpha=0.5)
        ax.set_title(title, fontsize=fontsize)
        if path:
            fig.savefig(path)
        if is_visible:
            plt.show()
        else:
            plt.close()
        return ax
