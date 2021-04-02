import pytest
import numpy as np
from copy import copy

from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.ets import ETSMAP, ETSFull, ETSAggregated
from orbit.constants.constants import PredictedComponents


@pytest.mark.parametrize("model_class", [ETSMAP, ETSFull, ETSAggregated])
def test_base_ets_init(model_class):
    ets = model_class()

    is_fitted = ets.is_fitted()

    model_data_input = ets._get_model_data_input()
    model_param_names = ets._get_model_param_names()
    init_values = ets._get_init_values()

    # model is not yet fitted
    assert not is_fitted
    # should only be initialized and not set
    assert not model_data_input
    # model param names should already be set
    assert model_param_names
    # callable is not implemented yet
    assert not init_values


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_ets_full_seasonal_fit(synthetic_data, estimator_type):
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
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_ets_aggregated_seasonal_fit(synthetic_data, estimator_type):
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

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP])
def test_ets_map_seasonal_fit(synthetic_data, estimator_type):
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

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
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
def test_full_prediction_percentiles(iclaims_training_data, prediction_percentiles):
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


# TODO: consider testing non-symmetric input percentiles
@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_map_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    ets = ETSMAP(
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
    # test behaviors without decomposition
    predicted_df = ets.predict(df)
    expected_columns = ['week'] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    # test behaviors with decomposition
    predicted_df = ets.predict(df, decompose=True)
    predicted_components = [
        'prediction',
        PredictedComponents.TREND.value,
        PredictedComponents.SEASONALITY.value]
    expected_columns = ['week']
    for pc in predicted_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize("seasonality", [1, 52])
def test_ets_full_reproducibility(synthetic_data, estimator_type, seasonality):
    train_df, test_df, coef = synthetic_data

    ets1 = ETSFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    # first fit and predict
    ets1.fit(train_df)
    posteriors1 = copy(ets1._posterior_samples)
    prediction1 = ets1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets2 = ETSFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    ets2.fit(train_df)
    posteriors2 = copy(ets2._posterior_samples)
    prediction2 = ets2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert prediction1.equals(prediction2)


@pytest.mark.parametrize("seasonality", [1, 52])
def test_ets_map_reproducibility(synthetic_data, seasonality):
    train_df, test_df, coef = synthetic_data

    ets1 = ETSMAP(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
    )

    # first fit and predict
    ets1.fit(train_df)
    posteriors1 = copy(ets1._aggregate_posteriors['map'])
    prediction1 = ets1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets2 = ETSMAP(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
    )

    ets2.fit(train_df)
    posteriors2 = copy(ets2._aggregate_posteriors['map'])
    prediction2 = ets2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert np.allclose(prediction1['prediction'].values, prediction2['prediction'].values)