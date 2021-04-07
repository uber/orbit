import pandas as pd
import numpy as np
from scipy.stats import norm

from ..constants import lm as constants
from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP
from .template import BaseTemplate, FullBayesianTemplate, MAPTemplate
from ..exceptions import ModelException
from ..utils.predictions import prepend_date_column


class LinearModel(BaseTemplate):
    """Linear Model

    Parameters
    ----------
    regressor_col : list
        Strings of names of regressors in dataset

    kwargs
        To specify `estimator_type` or additional args for the specified `estimator_type`
    """
    # data labels for sampler
    _data_input_mapper = constants.StanDataInput
    # used to match name of `*.stan` or `*.pyro` file to look for the model
    _model_name = "lm"

    def __init__(self, regressor_col, **kwargs):
        super().__init__(**kwargs)  # create estimator in base class
        self.regressor_col = regressor_col
        self.num_of_regressors = len(regressor_col)

        # Design matrix, depending on training data
        self.regressor_matrix = None

        # Model parameters to be sampling
        self.intercept = None
        self.coefficients = None
        self.obs_error_scale = None

    def _set_model_param_names(self):
        """Set posteriors keys to extract from sampling/optimization api"""
        self._model_param_names += [param.value for param in constants.StanSampleOutput]

    def _set_dynamic_attributes(self, df):
        """Set design matrix"""
        self._set_regressor_matrix(df)

    def _set_regressor_matrix(self, df):
        """Create design matrix given training data"""
        self._validate_df(df)
        self.regressor_matrix = np.zeros((self.num_of_observations, 0), dtype=np.double)
        if self.num_of_regressors > 0:
            self.regressor_matrix = df.filter(
                items=self.regressor_col,
            ).values

    def _validate_df(self, df):
        """Check if training dataframe containing the regressor columns"""
        df_columns = df.columns
        if self.regressor_col is not None and not set(self.regressor_col).issubset(
            df_columns
        ):
            raise ModelException(
                "DataFrame does not contain specified regressor colummn(s)."
            )

    def _predict(
        self, posterior_estimates, df, include_error=False, decompose=False, **kwargs
    ):
        """Create predictions for new covariates/regressors"""
        prediction_length = df.shape[0]

        intercept = posterior_estimates[
            constants.StanSampleOutput["INTERCEPT"].value
        ]
        coefficients = posterior_estimates[
            constants.StanSampleOutput["COEFFICIENTS"].value
        ]
        obs_error_scale = posterior_estimates[
            constants.StanSampleOutput["OBS_ERROR_SCALE"].value
        ]

        prediction_regressor_matrix = df.filter(
            items=self.regressor_col,
        ).values

        pred_sampling = np.stack(
            [
                np.random.normal(alpha + prediction_regressor_matrix @ beta, sigma)
                for alpha, beta, sigma in zip(intercept, coefficients, obs_error_scale)
            ],
            axis=0,
        )

        return {"prediction": pred_sampling}


class LinearModelFull(FullBayesianTemplate, LinearModel):
    _supported_estimator_types = [StanEstimatorMCMC]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LinearModelMAP(MAPTemplate, LinearModel):
    _supported_estimator_types = [StanEstimatorMAP]

    def __init__(self, estimator_type=StanEstimatorMAP, **kwargs):
        super().__init__(estimator_type=estimator_type, **kwargs)

    def predict(self, df, decompose=False, **kwargs):
        """Prediction intervals of Normal errors"""
        posterior_estimates = self._posterior_samples
        intercept = posterior_estimates[
            constants.StanSampleOutput["INTERCEPT"].value
        ]
        coefficients = posterior_estimates[
            constants.StanSampleOutput["COEFFICIENTS"].value
        ]
        obs_error_scale = posterior_estimates[
            constants.StanSampleOutput["OBS_ERROR_SCALE"].value
        ]

        prediction_regressor_matrix = df.filter(
            items=self.regressor_col,
        ).values
        expected_value = intercept + prediction_regressor_matrix @ coefficients

        prediction_dict = {'prediction' + "_" + str(p)
                           if p != 50 else 'prediction':
                           expected_value +
                           norm.ppf(0.01 * p) * obs_error_scale
                           for p in self._prediction_percentiles}
        prediction_df = pd.DataFrame(prediction_dict)
        prediction_df = prepend_date_column(prediction_df, df, self.date_col)
        return prediction_df
