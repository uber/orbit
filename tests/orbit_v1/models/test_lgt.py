import pytest
from orbit_v1.models.lgt import BaseLGT, LGTFull, LGTAggregated, LGTMAP
from orbit_v1.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP


def test_base_lgt_init():
    lgt = BaseLGT()

    is_fitted = lgt.is_fitted()

    stan_data_input = lgt._get_stan_data_input()
    model_param_names = lgt._get_model_param_names()
    stan_init = lgt._get_stan_init()

    assert not is_fitted  # model is not yet fitted
    assert not stan_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when stan_init callable is implemented
    assert not stan_init


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

    expected_columns = ['week', 5, 'prediction', 95]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

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


def test_lgt_map_univariate(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
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
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 11

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters