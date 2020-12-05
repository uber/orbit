import pytest
import numpy as np
import copy

from orbit.estimators.stan_estimator import StanEstimator, StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.ets import BaseETS, ETSFull, ETSAggregated, ETSMAP
from orbit.constants.constants import PredictedComponents


def test_base_ets_init():
    ets = BaseETS()

    is_fitted = ets.is_fitted()

    model_data_input = ets._get_model_data_input()
    model_param_names = ets._get_model_param_names()
    init_values = ets._get_init_values()

    assert not is_fitted  # model is not yet fitted
    assert not model_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when init_values callable is implemented
    assert not init_values


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_ets_full_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ets = ETSFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    ets.fit(train_df)
    predict_df = ets.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 6 # TODO: Check actual number

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_ets_aggregated_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ets = ETSAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    ets.fit(train_df)
    predict_df = ets.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 6 # TODO: Check Actual Number

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP])
def test_ets_map_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ets = ETSMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        verbose=False,
        estimator_type=estimator_type
    )

    ets.fit(train_df)
    predict_df = ets.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 5  # no `lp__` parameter in optimizing()

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_ets_non_seasonal_fit(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ets = ETSFull(
        response_col='response',
        date_col='week',
        estimator_type=estimator_type,
        num_warmup=50,
    )

    ets.fit(train_df)
    predict_df = ets.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 4

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    ets = ETSFull(
        response_col='claims',
        date_col='week',
        seasonality=52,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels = ['_5', '', '_95']
    else:
        p_labels = ['_5', '_10', '', '_95']

    ets.fit(df)
    predicted_df = ets.predict(df)
    expected_columns = ['week'] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    predicted_df = ets.predict(df, decompose=True)
    plot_components = [
        'prediction',
        PredictedComponents.TREND.value,
        PredictedComponents.SEASONALITY.value]

    expected_columns = ['week']
    for pc in plot_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize("seasonality", [1, 52])
def test_ets_full_reproducibility(synthetic_data, estimator_type, seasonality):
    train_df, test_df, coef = synthetic_data

    ets_first = ETSFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    # first fit and predict
    ets_first.fit(train_df)
    posteriors_first = copy.copy(ets_first._posterior_samples)
    predict_df_first = ets_first.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets_second = ETSFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    ets_second.fit(train_df)
    posteriors_second = copy.copy(ets_second._posterior_samples)
    predict_df_second = ets_second.predict(test_df)

    # assert same posterior keys
    assert set(posteriors_first.keys()) == set(posteriors_second.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors_first.items():
        assert np.allclose(posteriors_first[k], posteriors_second[k])

    # assert prediction is reproducible
    assert all(predict_df_first == predict_df_second)
