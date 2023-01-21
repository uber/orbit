import pytest
import numpy as np
from copy import copy

from orbit.models import ETS
from orbit.constants.constants import PredictionKeys


@pytest.mark.parametrize("estimator", ["stan-map", "stan-mcmc"])
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
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator="stan-mcmc",
    )
    ets.fit(train_df)

    init_values = ets._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    predict_df = ets.predict(test_df, decompose=False)

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(ets._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("point_method", ["mean", "median"])
def test_ets_aggregated_seasonal_fit(make_weekly_data, point_method):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col="response",
        date_col="week",
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator="stan-mcmc",
        n_bootstrap_draws=1e4,
    )
    ets.fit(train_df, point_method=point_method)

    init_values = ets._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    predict_df = ets.predict(test_df, decompose=False)

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(ets._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("n_bootstrap_draws", [None, -1, 1e4])
def test_ets_map_seasonal_fit(make_weekly_data, n_bootstrap_draws):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col="response",
        date_col="week",
        seasonality=52,
        verbose=False,
        estimator="stan-map",
        n_bootstrap_draws=n_bootstrap_draws,
    )

    ets.fit(train_df)
    init_values = ets._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    predict_df = ets.predict(test_df)

    # if n_bootstrap_draws provided then produce prediction range
    if n_bootstrap_draws is not None and n_bootstrap_draws > 0:
        expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    else:
        expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(ets._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    p = ets.get_posterior_samples()
    assert len(p.keys()) == expected_num_parameters


def test_ets_non_seasonal_fit(make_weekly_data):
    train_df, test_df, coef = make_weekly_data

    ets = ETS(
        response_col="response",
        date_col="week",
        num_warmup=50,
        num_sample=50,
    )
    ets.fit(train_df)
    predict_df = ets.predict(test_df)

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(ets._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ets._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_full_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    ets = ETS(
        response_col="claims",
        date_col="week",
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels = ["_5", "", "_95"]
    else:
        p_labels = ["_5", "_10", "", "_95"]

    # test behaviors without decomposition
    ets.fit(df)
    predicted_df = ets.predict(df)
    expected_columns = ["week"] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    # test behaviors with decomposition
    predicted_df = ets.predict(df, decompose=True)
    plot_components = [
        "prediction",
        PredictionKeys.TREND.value,
        PredictionKeys.SEASONALITY.value,
    ]

    expected_columns = ["week"]
    for pc in plot_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("n_bootstrap_draws", [-1, 1e4])
@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_map_prediction_percentiles(
    iclaims_training_data, n_bootstrap_draws, prediction_percentiles
):
    df = iclaims_training_data

    ets = ETS(
        response_col="claims",
        date_col="week",
        seasonality=52,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
        estimator="stan-map",
        n_bootstrap_draws=n_bootstrap_draws,
    )

    if n_bootstrap_draws > 0:
        if not prediction_percentiles:
            p_labels = ["_5", "", "_95"]
        else:
            p_labels = ["_5", "_10", "", "_95"]
    else:
        p_labels = [""]

    ets.fit(df)
    # test behaviors without decomposition
    predicted_df = ets.predict(df)

    expected_columns = ["week"] + ["prediction" + p for p in p_labels]

    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    # test behaviors with decomposition
    predicted_df = ets.predict(df, decompose=True)
    predicted_components = [
        "prediction",
        PredictionKeys.TREND.value,
        PredictionKeys.SEASONALITY.value,
    ]
    expected_columns = ["week"]

    for pc in predicted_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map"])
def test_ets_missing(iclaims_training_data, estimator):
    df = iclaims_training_data
    missing_idx = np.array([10, 20, 30, 40, 41, 42, 43, 44, df.shape[0] - 1])
    df.loc[missing_idx, "claims"] = np.nan

    dlt = ETS(
        response_col="claims",
        date_col="week",
        seasonality=52,
        verbose=False,
        estimator=estimator,
    )

    dlt.fit(df)
    predicted_df = dlt.predict(df)
    if estimator == "stan-map":
        expected_columns = ["week", "prediction"]
    elif estimator == "stan-mcmc":
        expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]

    assert all(~np.isnan(predicted_df["prediction"]))
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("seasonality", [1, 52])
def test_ets_full_reproducibility(make_weekly_data, seasonality):
    train_df, test_df, coef = make_weekly_data

    ets1 = ETS(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        num_sample=50,
        estimator="stan-mcmc",
    )

    # first fit and predict
    ets1.fit(train_df)
    posteriors1 = ets1.get_posterior_samples()
    prediction1 = ets1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets2 = ETS(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        num_sample=50,
        estimator="stan-mcmc",
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
        response_col="response",
        date_col="week",
        seasonality=seasonality,
        estimator="stan-map",
    )

    # first fit and predict
    ets1.fit(train_df)
    posteriors1 = ets1.get_point_posteriors()["map"]
    prediction1 = ets1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    ets2 = ETS(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        estimator="stan-map",
    )

    ets2.fit(train_df)
    posteriors2 = ets2.get_point_posteriors()["map"]
    prediction2 = ets2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert np.allclose(
        prediction1["prediction"].values, prediction2["prediction"].values
    )


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map"])
@pytest.mark.parametrize("random_seed", [10, 100])
def test_ets_predict_seed(make_weekly_data, estimator, random_seed):
    train_df, test_df, coef = make_weekly_data
    args = {
        "response_col": "response",
        "date_col": "week",
        "seasonality": 52,
        "n_bootstrap_draws": 100,
        "verbose": False,
        "estimator": estimator,
    }

    if estimator == "stan-mcmc":
        args.update({"num_warmup": 50, "num_sample": 100})
    elif estimator == "pyro-svi":
        args.update({"num_steps": 10})

    ets = ETS(**args)
    ets.fit(train_df)
    predict_df1 = ets.predict(test_df, seed=random_seed)
    predict_df2 = ets.predict(test_df, seed=random_seed)

    assert all(predict_df1["prediction"].values == predict_df2["prediction"].values)


@pytest.mark.parametrize(
    "idx_range",
    [
        [0, 100],
        [0, 50],
        [50, 100],
        [50, 80],
        [50, 120],
        [100, 150],
    ],
    ids=[
        "train-start-to-train-end",
        "train-start-to-middle",
        "middle-to-train-end",
        "train-period-subset",
        "train-test-period-cross",
        "completely-test-period",
    ],
)
def test_ets_predict_range(make_weekly_data, idx_range):
    train_cut_off = 100
    base_df, _, _ = make_weekly_data
    train_df = base_df[:train_cut_off].reset_index(drop=True)
    predict_df = base_df[idx_range[0] : idx_range[1]].reset_index(drop=True)

    ets = ETS(
        response_col="response",
        date_col="week",
        seasonality=52,
        verbose=False,
        estimator="stan-map",
    )
    ets.fit(train_df)
    ets.predict(predict_df)
