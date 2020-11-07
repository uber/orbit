import pandas as pd
import numpy as np
import torch
from copy import copy, deepcopy

from ..estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from ..exceptions import IllegalArgument, ModelException, PredictionException
from .base_model import BaseModel


class BaseLinearRegression(BaseModel):
    def __init__(self, response_col, regressor_col,
                 egressor_beta_prior=None, regressor_sigma_prior=None,
                 regression_penalty='fixed_ridge', lasso_scale=0.5, auto_ridge_scale=0.5):
        super().__init__()


class LinearRegressionFull(BaseLinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

