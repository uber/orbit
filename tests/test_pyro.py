from enum import Enum
import pytest
from orbit.lgt import LGT
from orbit.exceptions import IllegalArgument, EstimatorException


@pytest.mark.parametrize("sample_method,predict_method", [
    ("map", "map"),
    ("vi", "full"),
])
def test_fit_and_predict_univariate(
        synthetic_data, sample_method, predict_method):
    train_df, test_df, coef = synthetic_data
    lgt = LGT(
        response_col='response',
        date_col='week',
        seasonality=52,
        sample_method=sample_method,
        predict_method=predict_method,
        inference_engine='pyro',
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    # assert number of posterior param keys
    assert len(lgt.posterior_samples) == 13

    # assert output shape
    if predict_method == 'full':
        expected_columns = ['week', 5, 50, 95]
        expected_shape = (51, len(expected_columns))
        assert predict_df.shape == expected_shape
        assert predict_df.columns.tolist() == expected_columns
    else:
        expected_columns = ['week', 'prediction']
        expected_shape = (51, len(expected_columns))
        assert predict_df.shape == expected_shape
        assert predict_df.columns.tolist() == expected_columns


@pytest.mark.parametrize("sample_method,predict_method", [
    ("map", "map"),
    ("vi", "full"),
])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_fit_and_predict_with_regression(
        sample_method, predict_method, synthetic_data, regressor_signs):
    train_df, test_df, coef = synthetic_data

    lgt = LGT(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        seasonality=52,
        sample_method=sample_method,
        predict_method=predict_method,
        inference_engine='pyro',
        pyro_map_args={'num_steps': 31, 'learning_rate': 0.1},
        pyro_vi_args={'num_steps': 31, 'learning_rate': 0.1},
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    num_regressors = lgt.get_regression_coefs().shape[0]

    assert num_regressors == len(train_df.columns.tolist()[2:])

    # assert output shape
    if predict_method == 'full':
        expected_columns = ['week', 5, 50, 95]
        expected_shape = (51, len(expected_columns))
        assert predict_df.shape == expected_shape
        assert predict_df.columns.tolist() == expected_columns
    else:
        expected_columns = ['week', 'prediction']
        expected_shape = (51, len(expected_columns))
        assert predict_df.shape == expected_shape
        assert predict_df.columns.tolist() == expected_columns


def test_lgt_pyro_fit(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        inference_engine='pyro',
        sample_method='vi',
        predict_method='full',
    )

    lgt.fit(df=iclaims_training_data)
    expected_posterior_parameters = 13
    assert len(lgt.posterior_samples) == expected_posterior_parameters


def test_lgt_pyro_fit_with_missing_input(iclaims_training_data):
    class MockInputMapper(Enum):
        SOME_STAN_INPUT = 'some_stan_input'

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        inference_engine='pyro',
        sample_method='vi',
        predict_method='full',
    )

    lgt._data_input_mapper = MockInputMapper

    with pytest.raises(EstimatorException):
        lgt.fit(df=iclaims_training_data)


def test_lgt_pyro_fit_and_mean_predict(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        inference_engine='pyro',
        sample_method='vi',
        predict_method='mean',
    )

    lgt.fit(df=iclaims_training_data)

    predicted_df = lgt.predict(df=iclaims_training_data)

    expected_shape = (443, 2)
    expected_columns = ['week', 'prediction']

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_lgt_pyro_fit_and_full_predict(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        prediction_percentiles=[5, 95, 30],
        inference_engine='pyro',
        sample_method='vi',
        predict_method='full',
    )

    lgt.fit(df=iclaims_training_data)

    predicted_out = lgt.predict(df=iclaims_training_data)

    expected_columns = ['week', 5, 30, 50, 95]
    expected_shape = (443, len(expected_columns))

    assert predicted_out.shape == expected_shape
    assert list(predicted_out.columns) == expected_columns
