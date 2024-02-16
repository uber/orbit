import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from enum import Enum
import math
from scipy.stats import nct
import matplotlib.pyplot as plt

from ..constants.constants import (
    PredictionKeys,
    TrainingMetaKeys,
    PredictionMetaKeys,
)
from ..constants.palette import OrbitPalette
from ..exceptions import IllegalArgument, ModelException
from .model_template import ModelTemplate
from ..estimators.stan_estimator import StanEstimatorMAP
from ..utils.kernels import sandwich_kernel
from ..utils.features import make_fourier_series_df
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
    _DEGREE_OF_FREEDOM = "DOF"
    # ----------  Level  ---------- #
    NUM_KNOTS_LEVEL = "N_KNOTS_LEV"
    LEVEL_KNOT_SCALE = "LEV_KNOT_SCALE"
    KERNEL_LEVEL = "K_LEV"
    # ----------  Regression  ---------- #
    NUM_KNOTS_COEFFICIENTS = "N_KNOTS_COEF"
    KERNEL_COEFFICIENTS = "K_COEF"
    NUM_OF_REGRESSORS = "P"
    REGRESSOR_MATRIX = "REGRESSORS"
    COEFFICIENTS_INITIAL_KNOT_SCALE = "COEF_INIT_KNOT_SCALE"
    COEFFICIENTS_KNOT_SCALE = "COEF_KNOT_SCALE"


class BaseSamplingParameters(Enum):
    """
    The output sampling parameters related with base model.
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
    COEFFICIENTS = "coef"


# class KTRLiteInitializer(object):
#     def __init__(self, num_regressor, num_knots_coefficients):
#         self.num_regressor = num_regressor
#         self.num_knots_coefficients = num_knots_coefficients

#     def __call__(self):
#         init_values = dict()
#         if self.num_regressor > 1:
#             init_values[
#                 RegressionSamplingParameters.COEFFICIENTS_KNOT.value
#             ] = np.zeros((self.num_regressor, self.num_knots_coefficients))
#         return init_values


class KTRLiteModel(ModelTemplate):
    """
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
        the number of segments partitioned by the knots of seasonality; when it is 0, returns to a static coefficients
        of regression used in the seasonality estimation
    seasonal_initial_knot_scale : float
        scale parameter for seasonal regressors initial coefficient knots; default to be 1
    seasonal_knot_scale : float
        scale parameter for seasonal regressors drift of coefficient knots; default to be 0.1.
    degree_of_freedom : int
        degree of freedom for error t-distribution
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    """

    _data_input_mapper = DataInputMapper
    _model_name = "ktrlite"
    _supported_estimator_types = [StanEstimatorMAP]

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
        seasonality_segments=0,
        seasonal_initial_knot_scale=1.0,
        seasonal_knot_scale=0.1,
        degree_of_freedom=30,
        date_freq=None,
        **kwargs,
    ):
        # estimator is created in base class
        super().__init__(**kwargs)
        self.level_knot_scale = level_knot_scale
        self.level_segments = level_segments
        self.level_knot_dates = level_knot_dates
        self.level_knot_distance = level_knot_distance

        self.seasonality = seasonality
        self.seasonality_fs_order = seasonality_fs_order
        self.seasonality_segments = seasonality_segments
        self.seasonal_initial_knot_scale = seasonal_initial_knot_scale
        self.seasonal_knot_scale = seasonal_knot_scale

        # set private var to arg value
        # if None set default in _set_default_args()
        # use public one if knots length is not available
        self._seasonality = self.seasonality
        self._seasonality_fs_order = self.seasonality_fs_order
        self._seasonal_initial_knot_scale = None
        self._seasonal_knot_scale = None

        self._level_knot_dates = self.level_knot_dates
        self._degree_of_freedom = degree_of_freedom

        self.date_freq = date_freq

        # regression attributes -- used for fourier series as seasonality only
        self.num_of_regressors = 0
        self.regressor_col = list()
        self.regressor_col_grp = list()
        self.coefficients_initial_knot_scale = list()
        self.coefficients_knot_scale = list()

        # set static data attributes
        self._set_default_args()
        self._set_seasonality_attributes()

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
        self._set_model_param_names()

    def set_init_values(self):
        """Override function from Base Template"""
        init_values = None
        if len(self._seasonality) > 1 and self.num_of_regressors > 0:
            init_values = dict()
            init_values[
                RegressionSamplingParameters.COEFFICIENTS_KNOT.value
            ] = np.zeros((self.num_of_regressors, self.num_knots_coefficients))
            self._init_values = init_values

    def _set_default_args(self):
        """Set default attributes for None"""
        if self.seasonality is None:
            self._seasonality = list()
            self._seasonality_fs_order = list()
        elif not isinstance(self._seasonality, list) and isinstance(
            self._seasonality, (int, float)
        ):
            self._seasonality = [self.seasonality]

        # set some defaults for seasonality_fs_order
        if self._seasonality and self._seasonality_fs_order is None:
            self._seasonality_fs_order = [2] * len(self._seasonality)
        elif not isinstance(self._seasonality_fs_order, list) and isinstance(
            self._seasonality_fs_order, (int, float)
        ):
            self._seasonality_fs_order = [self.seasonality_fs_order]

        if len(self._seasonality_fs_order) != len(self._seasonality):
            raise IllegalArgument("length of seasonality and fs_order not matching")

        for k, order in enumerate(self._seasonality_fs_order):
            if 2 * order > self._seasonality[k] - 1:
                raise IllegalArgument(
                    "reduce seasonality_fs_order to avoid over-fitting"
                )

        if not isinstance(self.seasonal_initial_knot_scale, list) and isinstance(
            self.seasonal_initial_knot_scale * 1.0, float
        ):
            self._seasonal_initial_knot_scale = [
                self.seasonal_initial_knot_scale
            ] * len(self._seasonality)
        else:
            self._seasonal_initial_knot_scale = self.seasonal_initial_knot_scale

        if not isinstance(self.seasonal_knot_scale, list) and isinstance(
            self.seasonal_knot_scale * 1.0, float
        ):
            self._seasonal_knot_scale = [self.seasonal_knot_scale] * len(
                self._seasonality
            )
        else:
            self._seasonal_knot_scale = self.seasonal_knot_scale

    def _set_seasonality_attributes(self):
        """given list of seasonalities and their order, create list of seasonal_regressors_columns"""
        self.regressor_col_grp = list()
        self.regressor_col = list()
        self.coefficients_initial_knot_scale = list()
        self.coefficients_knot_scale = list()

        if len(self._seasonality) > 0:
            for idx, s in enumerate(self._seasonality):
                fs_cols = []
                order = self._seasonality_fs_order[idx]
                self.coefficients_initial_knot_scale += (
                    [self._seasonal_initial_knot_scale[idx]] * order * 2
                )
                self.coefficients_knot_scale += (
                    [self._seasonal_knot_scale[idx]] * order * 2
                )
                for i in range(1, order + 1):
                    fs_cols.append("seas{}_fs_cos{}".format(s, i))
                    fs_cols.append("seas{}_fs_sin{}".format(s, i))
                # flatten version of regressor columns
                self.regressor_col += fs_cols
                # list of group of regressor columns bundled with seasonality
                self.regressor_col_grp.append(fs_cols)

        self.num_of_regressors = len(self.regressor_col)

    # fit and predict related modules
    def _set_validate_ktr_params(self, training_meta):
        # avoid lengthy code
        response = training_meta[TrainingMetaKeys.RESPONSE.value]
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]

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
                df, _ = make_fourier_series_df(
                    df, s, order=order, prefix="seas{}_".format(s), shift=shift
                )

        return df

    def _set_regressor_matrix(self, df, training_meta):
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        # init of regression matrix depends on length of response vector
        self.regressor_matrix = np.zeros((num_of_observations, 0), dtype=np.double)
        if self.num_of_regressors > 0:
            self.regressor_matrix = df.filter(
                items=self.regressor_col,
            ).values

    def _set_kernel_matrix(self, df, training_meta):
        """set all kernel related metrics
        1. knot index (knot_idx)
        2. knot time point (knot_tp)
        3. kernel matrix (kernel)
        4. knot dates (knot_dates) based on frequency implied from df
        5. do the same on regression (seasonality regression) if regressors are available

        Parameters
        ----------
        df : pd.DataFrame
        training_meta : dict
        """
        num_of_observations = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        date_array = training_meta[TrainingMetaKeys.DATE_ARRAY.value]

        self._level_knots_idx = get_knot_idx(
            num_of_obs=num_of_observations,
            knot_distance=self.level_knot_distance,
            num_of_segments=self.level_segments,
            date_array=date_array,
            knot_dates=self.level_knot_dates,
        )
        # kernel of coefficients calculations
        # set some default
        self.kernel_coefficients = np.zeros((num_of_observations, 0), dtype=np.double)
        self.num_knots_coefficients = 0

        if self.num_of_regressors > 0:
            self._seas_knots_idx = get_knot_idx(
                date_array=None,
                num_of_obs=num_of_observations,
                knot_dates=None,
                knot_distance=None,
                num_of_segments=self.seasonality_segments,
            )

        tp = np.arange(1, num_of_observations + 1) / num_of_observations
        self.knots_tp_level = (1 + self._level_knots_idx) / num_of_observations
        self.kernel_level = sandwich_kernel(tp, self.knots_tp_level)
        self.num_knots_level = len(self.knots_tp_level)
        if self.date_freq is None:
            self.date_freq = pd.infer_freq(date_array)
            # self.time_delta = date_array.diff().min()
        self._level_knot_dates = get_knot_dates(
            date_array[0], self._level_knots_idx, self.date_freq
        )

        # update rest of the seasonality related fields
        if self.num_of_regressors > 0:
            self.knots_tp_coefficients = (
                1 + self._seas_knots_idx
            ) / num_of_observations
            self.kernel_coefficients = sandwich_kernel(tp, self.knots_tp_coefficients)
            self.num_knots_coefficients = len(self.knots_tp_coefficients)
            self._coef_knot_dates = get_knot_dates(
                date_array[0], self._seas_knots_idx, self.date_freq
            )

    def set_dynamic_attributes(self, df, training_meta):
        """Overriding the parent class to customize pre-processing in fitting process"""
        # extra settings and validation for KTRLite
        self._set_validate_ktr_params(training_meta)
        # attach fourier series as regressors
        df = self._make_seasonal_regressors(df, shift=0)
        # set regressors as input matrix and derive kernels
        self._set_regressor_matrix(df, training_meta)
        self._set_kernel_matrix(df, training_meta)

    def _set_model_param_names(self):
        """Model parameters to extract"""
        self._model_param_names += [param.value for param in BaseSamplingParameters]
        if len(self._seasonality) > 0 or self.num_of_regressors > 0:
            self._model_param_names += [
                param.value for param in RegressionSamplingParameters
            ]

    def predict(
        self,
        posterior_estimates,
        df,
        training_meta,
        prediction_meta,
        include_error=False,
        **kwargs,
    ):
        """Vectorized version of prediction math"""
        ################################################################
        # Prediction Attributes
        ################################################################
        start = prediction_meta[PredictionMetaKeys.START_INDEX.value]
        trained_len = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        output_len = prediction_meta[PredictionMetaKeys.PREDICTION_DF_LEN.value]

        ################################################################
        # Model Attributes
        ################################################################
        model = deepcopy(posterior_estimates)
        # TODO: adopt torch ?
        # for k, v in model.items():
        #     model[k] = torch.from_numpy(v)

        # We can pull any arbitrary value from the dictionary because we hold the
        # safe assumption: the length of the first dimension is always the number of samples
        # thus can be safely used to determine `num_sample`. If predict_method is anything
        # other than full, the value here should be 1
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        ################################################################
        # Trend Component
        ################################################################
        new_tp = np.arange(start + 1, start + output_len + 1) / trained_len
        if include_error:
            # in-sample knots
            lev_knot_in = model.get(BaseSamplingParameters.LEVEL_KNOT.value)
            # TODO: hacky way; let's just assume last two knot distance is knots distance for all knots
            lev_knot_width = self.knots_tp_level[-1] - self.knots_tp_level[-2]
            # check whether we need to put new knots for simulation
            if new_tp[-1] > 1:
                # derive knots tp
                if self.knots_tp_level[-1] + lev_knot_width >= new_tp[-1]:
                    knots_tp_level_out = np.array([new_tp[-1]])
                else:
                    knots_tp_level_out = np.arange(
                        self.knots_tp_level[-1] + lev_knot_width,
                        new_tp[-1],
                        lev_knot_width,
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

        trend = np.matmul(lev_knot, kernel_level.transpose(1, 0))

        ################################################################
        # Seasonality Component
        ################################################################
        # init of regression matrix depends on length of response vector
        total_seas_regression = np.zeros(trend.shape, dtype=np.double)
        seas_decomp = {}
        # update seasonal regression matrices
        if self._seasonality and self.regressor_col:
            df = self._make_seasonal_regressors(df, shift=start)
            coef_knot = model.get(RegressionSamplingParameters.COEFFICIENTS_KNOT.value)
            kernel_coefficients = sandwich_kernel(new_tp, self.knots_tp_coefficients)
            coef = np.matmul(coef_knot, kernel_coefficients.transpose(1, 0))
            pos = 0
            for idx, cols in enumerate(self.regressor_col_grp):
                seasonal_regressor_matrix = df[cols].values
                seas_coef = coef[..., pos : (pos + len(cols)), :]
                seas_regression = np.sum(
                    seas_coef * seasonal_regressor_matrix.transpose(1, 0), axis=-2
                )
                seas_decomp[
                    "seasonality_{}".format(self._seasonality[idx])
                ] = seas_regression
                pos += len(cols)
                total_seas_regression += seas_regression
        if include_error:
            epsilon = nct.rvs(
                self._degree_of_freedom,
                nc=0,
                loc=0,
                scale=obs_scale,
                size=(num_sample, len(new_tp)),
            )
            pred_array = trend + total_seas_regression + epsilon
        else:
            pred_array = trend + total_seas_regression

        out = {
            PredictionKeys.PREDICTION.value: pred_array,
            PredictionKeys.TREND.value: trend,
        }
        out.update(seas_decomp)

        return out

    # TODO: need a unit test of this function
    def get_level_knots(
        self, training_meta, point_method, point_posteriors, posterior_samples
    ):
        """Given posteriors, return knots and correspondent date"""
        date_col = training_meta[TrainingMetaKeys.DATE_COL.value]
        # since KTRLite only supports MAP estimator, point_method is guaranteed to be MAP
        lev_knots = point_posteriors.get(point_method).get(
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
        # since KTRLite only supports MAP estimator, point_method is guaranteed to be MAP
        levs = point_posteriors.get(point_method).get(
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
