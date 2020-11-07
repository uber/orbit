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


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_lgt_full_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    if issubclass(estimator_type, StanEstimator):
        lr = LinearRegressionFull(
            response_col='response',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            num_warmup=50,
            verbose=False,
            estimator_type=estimator_type
        )
    elif issubclass(estimator_type, PyroEstimator):
        lr = LinearRegressionFull(
            response_col='response',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            num_steps=10,
            verbose=False,
            estimator_type=estimator_type
        )

    lr.fit(train_df)
    predict_df = lr.predict(test_df)

    regression_out = lr.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])
