import pytest

from orbit.estimators.pyro_estimator import PyroEstimator, PyroEstimatorVI, PyroEstimatorMAP
from orbit.estimators.stan_estimator import StanEstimator, StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.lgt import BaseLGT, LGTFull, LGTAggregated, LGTMAP


def test_base_lgt_init():
    lgt = BaseLGT()

    is_fitted = lgt.is_fitted()

    model_data_input = lgt._get_model_data_input()
    model_param_names = lgt._get_model_param_names()
    init_values = lgt._get_init_values()

    assert not is_fitted  # model is not yet fitted
    assert not model_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when init_values callable is implemented
    assert not init_values


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_full_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_lgt_full_univariate_pyro(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_steps=10,
        verbose=False,
        estimator_type=PyroEstimatorVI
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` in pyro

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_aggregated_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_lgt_aggregated_univariate_pyro(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        verbose=False,
        num_steps=10,
        estimator_type=PyroEstimatorVI
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` in pyro

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP, PyroEstimatorMAP])
def test_lgt_map_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        verbose=False,
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` parameter in optimizing()

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_non_seasonal_fit(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        estimator_type=estimator_type,
        num_warmup=50,
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 11

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_lgt_non_seasonal_fit_pyro(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        estimator_type=PyroEstimatorVI,
        num_steps=10
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 10  # no `lp__` in pyro

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI])
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
        lgt = LGTFull(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_warmup=50,
            verbose=False,
            estimator_type=estimator_type
        )
    elif issubclass(estimator_type, PyroEstimator):
        lgt = LGTFull(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_steps=10,
            verbose=False,
            estimator_type=estimator_type
        )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction_lower', 'prediction', 'prediction_upper']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_lgt_aggregated_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    if issubclass(estimator_type, StanEstimator):
        lgt = LGTAggregated(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            num_warmup=50,
            verbose=False,
            estimator_type=estimator_type
        )
    elif issubclass(estimator_type, PyroEstimator):
        lgt = LGTAggregated(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            num_steps=10,
            verbose=False,
            estimator_type=estimator_type
        )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])
