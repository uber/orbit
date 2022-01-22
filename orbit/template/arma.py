

import numpy as np
import pandas as pd
from scipy.stats import nct
from copy import deepcopy
import torch
from enum import Enum

from ..constants.constants import (
    DEFAULT_REGRESSOR_SIGN,
    DEFAULT_REGRESSOR_BETA,
    DEFAULT_REGRESSOR_SIGMA,
    COEFFICIENT_DF_COLS,
    PredictMethod,
    PredictionKeys,
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
    AR_LAGS = 'LAG AR'
    MA_LAGS = 'LAG_MA'
    #NUM_OF_OBS = 'NUM_OF_OBS' 
    LM_FIRST = 'LM_FIRST'   



class BaseSamplingParameters(Enum):
    """
    base parameters in posteriors sampling
    """
    # ---------- Noise Trend ---------- #
    RESIDUAL_SIGMA = 'obs_sigma'
    # ---------- ARMA Model Specific ---------- #


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
        self.num_of_regressors =  num_of_regressors
 
    def __call__(self):
        init_values = dict()
        if self.num_of_ar_lags > 0:
            init_ar = np.clip(np.random.normal(loc=0, scale=1.0/self.num_of_ar_lags, size=self.num_of_ar_lags), -1.0, 1.0)
            init_values[LatentSamplingParameters.REGRESSION_AR_COEFFICIENTS.value] = init_ar
        if self.num_of_ma_lags > 0:
            init_ma = np.clip(np.random.normal(loc=0, scale=1.0/self.num_of_ma_lags, size=self.num_of_ma_lags), -1.0, 1.0)
            init_values[LatentSamplingParameters.REGRESSION_MA_COEFFICIENTS.value] = init_ma
        if self.num_of_regressors > 0:
            init_lm = np.clip(np.random.normal(loc=0, scale=1.0/self.num_of_regressors, size=self.num_of_regressors), -5.0, 5.0)
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
    #_fitter = None # not sure what this is 
    _supported_estimator_types = [StanEstimatorMAP, StanEstimatorMCMC]

    def __init__(self, num_of_ar_lags =0, num_of_ma_lags =0, lm_first = 0,  **kwargs):
        # set by ._set_init_values
        # this is ONLY used by stan which by default used 'random'
        super().__init__(**kwargs)
        self._init_values = None
        self.num_of_regressors = 0
        self.regessor_matrix = None

        # set by _set_model_param_names()
        self._rho = list() # AR 
        self._theta = list() # MA
        self._beta = list() # LM 
        self._mu = list() # mean
        
        # the arma stuff 
        self.num_of_ar_lags = num_of_ar_lags
        self.num_of_ma_lags = num_of_ma_lags
        self.ar_lags = list()
        self.ma_lags = list()

        self.lm_first = lm_first
        
        self._set_model_param_names()

        
    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api
        Notes
        -----
        Overriding :func: `~orbit.models.BaseETS._set_model_param_names`
        It sets additional required attributes related to trend and regression
        """
        print("here")
        self._model_param_names += [param.value for param in BaseSamplingParameters]


        # append regressors if any
        if self.num_of_regressors > 0:
            self._model_param_names += [
                RegressionSamplingParameters.REGRESSION_COEFFICIENTS.value]




    def set_init_values(self):
        """Optional; set init as a callable (for Stan ONLY)
        See: https://pystan.readthedocs.io/en/latest/api.htm
        """
        pass


    def predict(self, posterior_estimates, df, training_meta, prediction_meta, include_error=False, **kwargs):
    	# this is currently only going to use the mu 
        """Vectorized version of prediction math"""
        ################################################################
        # Prediction Attributes
        ################################################################
        n_forecast_steps = prediction_meta[PredictionMetaKeys.FUTURE_STEPS.value]
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


        # mu 
        regressor_mu = model.get(MUSamplingParameters.SIGNAL_MU.value)
        # ar 
        regressor_rho = model.get(ARSamplingParameters.AR_RHO.value)
        # ma 
        regressor_theta = model.get(MASamplingParameters.MA_THETA.value)
        # lm
        regressor_beta = model.get(LMSamplingParameters.LM_BETA.value)


        ################################################################
        # Regression Component
        ################################################################
        # calculate regression component
        if self.regressor_col is not None and len(self.regressor_col) > 0:
            regressor_matrix = df[self._regressor_col].values
            if not np.all(np.isfinite(regressor_matrix)):
                raise PredictionException("Invalid regressors values. They must be all not missing and finite.")
            regressor_beta = regressor_beta.t()
            if len(regressor_beta.shape) == 1:
                regressor_beta = regressor_beta.unsqueeze(0)
            regressor_torch = torch.from_numpy(regressor_matrix).double()
            regression = torch.matmul(regressor_torch, regressor_beta)
            regression = regression.t()
        else:
            # regressor is always dependent with df. hence, no need to make full size
            regression = torch.zeros((num_sample, output_len), dtype=torch.double)

 

        ################################################################
        # Combine Components
        ################################################################
        # this needs to be updated to include all parts 
        # trim component with right start index

        # sum components
        pred_array = regression

        pred_array = pred_array.numpy()
        regression = regression.numpy()

        out = {
            PredictionKeys.PREDICTION.value: pred_array,
            PredictionKeys.REGRESSION.value: regression
        }

        return out

    def get_regression_coefs(self, training_meta, point_method, point_posteriors, posterior_samples,
                             include_ci=False, lower=0.05, upper=0.95):
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

        # note ordering here is not the same as `self.regressor_cols` because regressors here are grouped by signs
        regressor_cols = lm



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
        
        
        
    def get_init_values(self):
        return self._init_values

    def get_model_param_names(self):
        return self._model_param_names

    def get_data_input_mapper(self):
        return self._data_input_mapper

    def get_model_name(self):
        return self._model_name

    def get_fitter(self):
        return self._fitter

    def get_supported_estimator_types(self):
        return self._supported_estimator_types
    
    
    