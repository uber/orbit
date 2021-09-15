import pytest
import numpy as np
from copy import copy

from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP
from orbit.models import ETS
from orbit.template.ets import ETSInitializer
from orbit.constants.constants import PredictionKeys


@pytest.mark.parametrize("estimator", ['stan-map', 'stan-mcmc'])
def test_base_ets_init(estimator):
    ets = ETS(estimator=estimator)

    is_fitted = ets.is_fitted()

    model_data_input = ets.get_training_data_input()
    model_param_names = ets._model.get_model_param_names()
    init_values = ets._model.get_init_values()

    # model is not yet fitted
    assert not is_fitted
    # should only be initialized and not set
    assert not model_data_input
    # model param names should already be set
    assert model_param_names
    # callable is not implemented yet
    assert not init_values


def test_ets_full_seasonal_fit(make_weekly_data):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator='stan-mcmc',
    )
    ets.fit(train_df)

    init_call = ets._model.get_init_values()
    assert isinstance(init_call, ETSInitializer)
    assert init_call.s == 52
    init_values = init_call()
    assert init_values['init_sea'].shape == (51, )

    predict_df = ets.predict(test_df, decompose=False)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 5

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("point_method", ['mean', 'median'])
def test_ets_aggregated_seasonal_fit(make_weekly_data, point_method):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator='stan-mcmc',
        n_bootstrap_draws=1e4,
    )
    ets.fit(train_df, point_method=point_method)

    init_call = ets._model.get_init_values()
    assert isinstance(init_call, ETSInitializer)
    assert init_call.s == 52
    init_values = init_call()
    assert init_values['init_sea'].shape == (51, )

    predict_df = ets.predict(test_df, decompose=False)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 5

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("n_bootstrap_draws", [None, -1, 1e4])
def test_ets_map_seasonal_fit(make_weekly_data, n_bootstrap_draws):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator='stan-map',
        n_bootstrap_draws=n_bootstrap_draws,
    )

    ets.fit(train_df)
    init_call = ets._model.get_init_values()
    assert isinstance(init_call, ETSInitializer)
    assert init_call.s == 52
    init_values = init_call()
    assert init_values['init_sea'].shape == (51, )

    predict_df = ets.predict(test_df)

    # if n_bootstrap_draws provided then produce prediction range
    if n_bootstrap_draws is not None and n_bootstrap_draws > 0:
        expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    else:
        expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 5

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC])
def test_ets_non_seasonal_fit(make_weekly_data, estimator_type):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col='response',
        date_col='week',
        num_warmup=50,
        num_sample=50,
    )
    ets.fit(train_df)
    predict_df = ets.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 3

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_full_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    ets = ETS(
        response_col='claims',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels = ['_5', '', '_95']
    else:
        p_labels = ['_5', '_10', '', '_95']

    # test behaviors without decomposition
    ets.fit(df)
    predicted_df = ets.predict(df)
    expected_columns = ['week'] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    # test behaviors with decomposition
    predicted_df = ets.predict(df, decompose=True)
    plot_components = [
        'prediction',
        PredictionKeys.TREND.value,
        PredictionKeys.SEASONALITY.value]

    expected_columns = ['week']
    for pc in plot_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("n_bootstrap_draws", [-1, 1e4])
@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_map_prediction_percentiles(iclaims_training_data, n_bootstrap_draws, prediction_percentiles):
    df = iclaims_training_data

    ets = ETS(
        response_col='claims',
        date_col='week',
        seasonality=52,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
        estimator='stan-map',
        n_bootstrap_draws=n_bootstrap_draws,
    )

    if n_bootstrap_draws > 0:
        if not prediction_percentiles:
            p_labels = ['_5', '', '_95']
        else:
            p_labels = ['_5', '_10', '', '_95']
    else:
        p_labels = ['']

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
        PredictionKeys.TREND.value,
        PredictionKeys.SEASONALITY.value]
    expected_columns = ['week']

    for pc in predicted_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("seasonality", [1, 52])
def test_ets_full_reproducibility(make_weekly_data, seasonality):
    train_df, test_df, coef = make_weekly_data

    ets1 = ETS(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        num_sample=50,
        estimator='stan-mcmc',
    )

    # first fit and predict
    ets1.fit(train_df)
    posteriors1 = ets1.get_posterior_samples()
    prediction1 = ets1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets2 = ETS(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        num_sample=50,
        estimator='stan-mcmc',
    )

    ets2.fit(train_df)
    posteriors2 = ets2.get_posterior_samples()
    prediction2 = ets2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert prediction1.equals(prediction2)


@pytest.mark.parametrize("seasonality", [1, 52])
def test_ets_map_reproducibility(make_weekly_data, seasonality):
    train_df, test_df, coef = make_weekly_data

    ets1 = ETS(
        response_col='response',
        date_col='week',
        seasonality=seasonality,
        estimator='stan-map',
    )

    # first fit and predict
    ets1.fit(train_df)
    posteriors1 = ets1.get_point_posteriors()['map']
    prediction1 = ets1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets2 = ETS(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        estimator='stan-map',
    )

    ets2.fit(train_df)
    posteriors2 = ets2.get_point_posteriors()['map']
    prediction2 = ets2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert np.allclose(prediction1['prediction'].values, prediction2['prediction'].values)
