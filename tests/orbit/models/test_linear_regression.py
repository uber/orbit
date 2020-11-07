import pytest

from orbit.estimators.stan_estimator import StanEstimator, StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.linear_regression import BaseLinearRegression, LinearRegressionFull


def test_base_lgt_init():
    lr = BaseLinearRegression()

    is_fitted = lr.is_fitted()

    model_data_input = lr._get_model_data_input()
    model_param_names = lr._get_model_param_names()

    assert not is_fitted  # model is not yet fitted
    assert not model_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set


