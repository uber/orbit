import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from enum import Enum

from ..constants.constants import (
    COEFFICIENT_DF_COLS,
    PredictMethod,
    TrainingMetaKeys,
    PredictionMetaKeys
)
from .model_template import ModelTemplate
from ..exceptions import IllegalArgument, ModelException, PredictionException
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP


class DataInputMapper(Enum):
    """
    mapping from object input to stan file
    """
    # ----------  Regressions ---------- #
    NUM_OF_REGRESSORS = 'K'
    REGRESSOR_MATRIX = 'X'
    NUM_OF_AR_LAGS = 'P'
    NUM_OF_MA_LAGS = 'Q'
    AR_LAGS = 'LAG_AR'
    MA_LAGS = 'LAG_MA'
    LM_FIRST = 'LM_FIRST'


class BaseSamplingParameters(Enum):
    """
    base parameters in posteriors sampling
    """
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'
    # ---------- ARMA Model Specific ---------- #
    SIGNAL_MU = 'mu'


class MUSamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """
    SIGNAL_MU = 'mu'


class LMSamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """
    LM_BETA = 'beta'


class ARSamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """
    AR_RHO = 'rho'


class MASamplingParameters(Enum):
    """
    regression component related parameters in posteriors sampling
    """
    MA_THETA = 'theta'


class LatentSamplingParameters(Enum):
    """
    latent variables to be sampled
    """
    REGRESSION_AR_COEFFICIENTS = 'ar_rho'
    REGRESSION_MA_COEFFICIENTS = 'ma_theta'
    REGRESSION_LM_COEFFICIENTS = 'lm_beta'


# a callable object for generating initial values in sampling/optimization
class ARMAInitializer(object):
    def __init__(self, num_of_ar_lags, num_of_ma_lags, num_of_regressors):
        self.num_of_ar_lags = num_of_ar_lags
        self.num_of_ma_lags = num_of_ma_lags
        self.num_of_regressors = num_of_regressors

    def __call__(self):
        init_values = dict()
        if self.num_of_ar_lags > 0:
            init_ar = np.clip(np.random.normal(loc=0, scale=1.0 / self.num_of_ar_lags, size=self.num_of_ar_lags), -1.0,
                              1.0)
            init_values[LatentSamplingParameters.REGRESSION_AR_COEFFICIENTS.value] = init_ar
        if self.num_of_ma_lags > 0:
            init_ma = np.clip(np.random.normal(loc=0, scale=1.0 / self.num_of_ma_lags, size=self.num_of_ma_lags), -1.0,
                              1.0)
            init_values[LatentSamplingParameters.REGRESSION_MA_COEFFICIENTS.value] = init_ma
        if self.num_of_regressors > 0:
            init_lm = np.clip(np.random.normal(loc=0, scale=1.0 / self.num_of_regressors, size=self.num_of_regressors),
                              -5.0, 5.0)
            init_values[LatentSamplingParameters.REGRESSION_LM_COEFFICIENTS.value] = init_lm

        return init_values


class ARMAModel(ModelTemplate):
    """
    Notes
    -----
    contain data structure ; specify what need to fill from abstract to turn a model concrete
    """
    # class attributes
    _data_input_mapper = DataInputMapper
    _model_name = 'arma'
    # _fitter = None # not sure what this is
    _supported_estimator_types = [StanEstimatorMAP, StanEstimatorMCMC]

    def __init__(self, ar_lags, ma_lags, num_of_ma_lags, num_of_ar_lags, regressor_col, response_col, **kwargs):
        # set by ._set_init_values
        # this is ONLY used by stan which by default used 'random'
        super().__init__(**kwargs)
        self._init_values = None

        self.response_col = response_col
        self.regressor_col = None
        self.regessor_matrix = None
        self.num_of_regressors = 0
        if regressor_col is not None:
            self.num_of_regressors = len(regressor_col)
            self.regressor_col = regressor_col
            self.regessor_matrix = None

        # set by _set_model_param_names()
        self._rho = list()  # AR
        self._theta = list()  # MA
        self._beta = list()  # LM
        self._mu = list()  # mean

        # the arma stuff 
        self.num_of_ar_lags = num_of_ar_lags
        self.num_of_ma_lags = num_of_ma_lags
        self.ar_lags = np.array(ar_lags).astype(np.int64)
        self.ma_lags = np.array(ma_lags).astype(np.int64)
        self.lm_first = 0
        self._set_model_param_names()

    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api
        Notes
        -----
        Overriding :func: `~orbit.models.BaseETS._set_model_param_names`
        It sets additional required attributes related to trend and regression
        """
        self._model_param_names += [param.value for param in BaseSamplingParameters]

        # append ar if any
        if self.num_of_ar_lags > 0:
            self._model_param_names += [
                ARSamplingParameters.AR_RHO.value]

        # append ma if any
        if self.num_of_ma_lags > 0:
            self._model_param_names += [
                MASamplingParameters.MA_THETA.value]

        # append regressors if any
        if self.num_of_regressors > 0:
            self._model_param_names += [
                LMSamplingParameters.LM_BETA.value]

    def predict(self, posterior_estimates, df, training_meta, prediction_meta, include_error=False, **kwargs):
        # this is currently only going to use the mu
        """Vectorized version of prediction math"""
        ################################################################
        # Prediction Attributes
        ################################################################
        n_forecast_steps = prediction_meta[PredictionMetaKeys.FUTURE_STEPS.value]
        # this might need to always be the start of the data
        start = prediction_meta[PredictionMetaKeys.START_INDEX.value]
        trained_len = training_meta[TrainingMetaKeys.NUM_OF_OBS.value]
        output_len = prediction_meta[PredictionMetaKeys.PREDICTION_DF_LEN.value]
        full_len = trained_len + n_forecast_steps

        ################################################################
        # Model Attributes
        ################################################################
        model = deepcopy(posterior_estimates)
        for k, v in model.items():
            model[k] = torch.from_numpy(v)

        # We can pull any arbitrary value from the dictionary because we hold the
        # safe assumption: the length of the first dimension is always the number of samples
        # thus can be safely used to determine `num_sample`. If predict_method is anything
        # other than full, the value here should be 1
        arbitrary_posterior_value = list(model.values())[0]
        num_sample = arbitrary_posterior_value.shape[0]

        # mu series mean / trend
        regressor_mu = model.get(MUSamplingParameters.SIGNAL_MU.value)
        # auto regressor
        regressor_rho = model.get(ARSamplingParameters.AR_RHO.value)
        # moving average 
        regressor_theta = model.get(MASamplingParameters.MA_THETA.value)
        # linear model
        regressor_beta = model.get(LMSamplingParameters.LM_BETA.value)
        # the sigma
        residual_sigma = model.get(BaseSamplingParameters.RESIDUAL_SIGMA.value)

        ################################################################
        # mu Component; i.e., the trend 
        # this always happens; i.e., yhat = mu is the simplest possible model
        ################################################################
        # TODO: can't we just do torch.ones ?
        regressor_matrix = np.ones(full_len)

        regressor_mu = regressor_mu.unsqueeze(0)
        regressor_torch = torch.from_numpy(regressor_matrix).double().unsqueeze(0)
        pred_mu = torch.matmul(regressor_torch.t(), regressor_mu)
        pred_mu = pred_mu.t()

        ################################################################
        # random error prediction 
        ################################################################
        if include_error:
            error_value = np.random.normal(
                loc=0,
                scale=residual_sigma.unsqueeze(-1),
                size=pred_mu.shape)
        else:
            error_value = torch.zeros_like(residual_sigma)

        ################################################################
        # Regression Component
        ################################################################
        # calculate regression component
        if self.regressor_col is not None and self.num_of_regressors > 0:
            regressor_matrix = df[self.regressor_col].values
            if not np.all(np.isfinite(regressor_matrix)):
                raise PredictionException("Invalid regressors values. They must be all not missing and finite.")
            regressor_beta = regressor_beta.t()
            if len(regressor_beta.shape) == 1:
                regressor_beta = regressor_beta.unsqueeze(0)
            regressor_torch = torch.from_numpy(regressor_matrix).double()
            pred_lm = torch.matmul(regressor_torch, regressor_beta)
            pred_lm = pred_lm.t()
        else:
            # regressor is always dependent with df. hence, no need to make full size
            pred_lm = torch.zeros((num_sample, output_len), dtype=torch.double)

        ################################################################
        # ARMA terms definition:
        # resid: y - beta X
        # error: y - beta X - ar - ma (total error)
        ################################################################
        # this is the prediction so far 
        # there is S (number of MCMC samples) by N (Number of predictions that are made )
        pred_mu_lm = pred_mu + pred_lm

        # initialize the is the error used in the MA
        error = torch.zeros((num_sample, output_len), dtype=torch.double)
        # the observed data torch.tensor(df[self.response_col].values.copy())
        obs = torch.tile(torch.tensor(df[self.response_col].values.copy()), [num_sample, 1])

        if self.lm_first:  # r = y - beta X
            resid = obs - pred_mu_lm
        else:  # r = y
            resid = obs

        ################################################################
        # ARMA prediction 
        ################################################################
        # make the prediction arrays 
        pred_ar = torch.zeros((num_sample, output_len), dtype=torch.double)
        pred_ma = torch.zeros((num_sample, output_len), dtype=torch.double)

        for i in range(output_len):
            if self.num_of_ar_lags > 0:  # ar process
                for p in range(self.num_of_ar_lags):
                    if self.ar_lags[p] < i:
                        pred_ar[:, i] = pred_ar[:, i] + regressor_rho[:, p] * resid[:, i - self.ar_lags[p]]
            if self.num_of_ma_lags > 0:  # ma process
                for q in range(self.num_of_ma_lags):
                    if self.ma_lags[q] < i:
                        pred_ma[:, i] = pred_ma[:, i] + regressor_theta[:, q] * error[:, i - self.ma_lags[q]]
                # update the error for the ma model 
                if self.lm_first:
                    error[:, i] = - pred_ar[:, i] - pred_ma[:, i]
                else:
                    error[:, i] = obs[:, i] - pred_mu_lm[:, i] - pred_ar[:, i] - pred_ma[:, i]

            # update the obs with the prediction in the forecast range 
            if i > trained_len:
                obs[:, i] = pred_mu_lm[:, i] + pred_ar[:, i] + pred_ma[:, i] + error_value[:, i]
                if self.lm_first:  # r = y - beta X
                    resid[:, i] = obs[:, i] - pred_mu_lm[:, i]
                    error[:, i] = - pred_ar[:, i] - pred_ma[:, i]
                else:  # r = y
                    resid[:, i] = obs[:, i]
                    error[:, i] = obs[:, i] - pred_mu_lm[:, i] - pred_ar[:, i] - pred_ma[:, i]

        ################################################################
        # Combine Components
        ################################################################
        # this needs to be updated to include all parts 
        # trim component with right start index

        # sum components
        pred_all = pred_mu_lm + pred_ar + pred_ma + error_value

        pred_all = pred_all.numpy()
        pred_lm = pred_lm.numpy()
        pred_ar = pred_ar.numpy()
        pred_ma = pred_ma.numpy()

        out = {
            'prediction': pred_all,
            'trend': pred_mu,
            'regression': pred_lm,
            'autoregressor': pred_ar,
            'moving-average': pred_ma,
            'residual-error': error_value
        }

        return out

    def get_regression_coefs(self, training_meta, point_method, point_posteriors, posterior_samples,
                             include_ci=False, lower=0.05, upper=0.95):
        print("----------------------")
        print("----------------------")
        print("-get_regression_coefs-")
        print("----------------------")
        print("----------------------")

        """Return DataFrame regression coefficients.
          If point_method is None when fitting, return the median of coefficients.

        Parameters
        -----------
        include_ci : bool
            if including the confidence intervals for the regression coefficients
        lower : float between (0, 1). default to be 0.05
            lower bound for the CI
        upper : float between (0, 1). default to be 0.95.
            upper bound for the CI

        Returns
        -------
        pandas data frame holding the regression coefficients
        """
        # init dataframe
        coef_df = pd.DataFrame()

        # end if no regressors
        if self.num_of_regressors == 0:
            return coef_df

        _point_method = point_method
        if point_method is None:
            _point_method = PredictMethod.MEDIAN.value

        coef_mu = point_posteriors \
            .get(_point_method) \
            .get(MUSamplingParameters.SIGNAL_MU.value)

        coef_lm = point_posteriors \
            .get(_point_method) \
            .get(LMSamplingParameters.LM_BETA.value)

        coef_ar = point_posteriors \
            .get(_point_method) \
            .get(ARSamplingParameters.AR_RHO.value)

        coef_ma = point_posteriors \
            .get(_point_method) \
            .get(MASamplingParameters.MA_THETA.value)

        # get column names
        lm_cols = self.regressor_col

        coef_df[COEFFICIENT_DF_COLS.REGRESSOR] = regressor_cols
        coef_df[COEFFICIENT_DF_COLS.COEFFICIENT] = coef.flatten()

        # if we have posteriors distribution and also include ci
        if point_method is None and include_ci:
            coef_samples = posterior_samples.get(LMSamplingParameters.LM_BETA.value)
            coef_lower = np.quantile(coef_samples, lower, axis=0)
            coef_upper = np.quantile(coef_samples, upper, axis=0)
            coef_df_lower = coef_df.copy()
            coef_df_upper = coef_df.copy()
            coef_df_lower[COEFFICIENT_DF_COLS.COEFFICIENT] = coef_lower
            coef_df_upper[COEFFICIENT_DF_COLS.COEFFICIENT] = coef_upper

            return coef_df, coef_df_lower, coef_df_upper
        else:
            return coef_df

    # repeat for AR MA 
    def _set_regressor_matrix(self, df, num_of_observations):
        """Set regressor matrix based on the input data-frame.
        Notes
        -----
        In case of absence of regression, they will be set to np.array with dim (num_of_obs, 0) to fit Stan requirement
        """
        # init of regression matrix depends on length of response vector
        self.regressor_matrix = np.zeros((num_of_observations, 0), dtype=np.double)

        # update regression matrices
        if self.num_of_regressors > 0:
            self.regressor_matrix = df.filter(
                items=self.regressor_col, ).values
            if not np.all(np.isfinite(self.regressor_matrix)):
                raise ModelException("Invalid regressors values. They must be all not missing and finite.")

    def set_dynamic_attributes(self, df, training_meta):
        """Overriding: func: `~orbit.models.BaseETS._set_dynamic_attributes"""
        super().set_dynamic_attributes(df, training_meta)
        # depends on num_of_observations
        self._set_regressor_matrix(df, training_meta[TrainingMetaKeys.NUM_OF_OBS.value])

    def set_init_values(self):
        """Override function from Base Template
        """
        # partialfunc does not work when passed to PyStan because PyStan uses
        # inspect.getargspec(func) which seems to raise an exception with keyword-only args
        # caused by using partialfunc
        # lambda does not work in serialization in pickle
        # callable object as an alternative workaround
        init_values_callable = ARMAInitializer(
            self.num_of_ar_lags,
            self.num_of_ma_lags,
            self.num_of_regressors
        )
        self._init_values = init_values_callable
