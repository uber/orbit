import pytest
import numpy as np
import pandas as pd

from orbit.models import KTRLite
from orbit.diagnostics.metrics import smape

# used for in-sample training insanity check
SMAPE_TOLERANCE = 0.28


@pytest.mark.parametrize(
    "seasonality_fs_order", [None, [5]], ids=["default_order", "manual_order"]
)
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "single", "with_coef": False})], indirect=True
)
def test_ktrlite_single_seas(make_daily_data, seasonality_fs_order):
    train_df, _, _ = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        seasonality=[365.25],
        seasonality_fs_order=seasonality_fs_order,
        estimator="stan-map",
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(train_df)

    expected_columns = ["date", "prediction"]
    expected_shape = (train_df.shape[0], len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    smape_val = smape(train_df["response"].values, predict_df["prediction"].values)
    assert smape_val <= SMAPE_TOLERANCE


@pytest.mark.parametrize(
    "seasonality_fs_order", [None, [2, 5]], ids=["default_order", "manual_order"]
)
@pytest.mark.parametrize(
    "make_daily_data", [({"with_dual_sea": True, "with_coef": False})], indirect=True
)
def test_ktrlite_dual_seas(make_daily_data, seasonality_fs_order):
    train_df, _, _ = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=seasonality_fs_order,
        estimator="stan-map",
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(train_df)

    expected_columns = ["date", "prediction"]
    expected_shape = (train_df.shape[0], len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    smape_val = smape(train_df["response"].values, predict_df["prediction"].values)
    assert smape_val <= SMAPE_TOLERANCE


@pytest.mark.parametrize(
    "make_daily_data", [({"with_dual_sea": True, "with_coef": False})], indirect=True
)
@pytest.mark.parametrize("level_segments", [20, 10, 2])
def test_ktrlite_level_segments(make_daily_data, level_segments):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        level_segments=level_segments,
        estimator="stan-map",
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ["date", "prediction"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    knots_df = ktrlite.get_level_knots()
    levels_df = ktrlite.get_levels()
    assert knots_df.shape[0] in [level_segments + 1, level_segments + 2]
    assert levels_df.shape[0] == ktrlite.get_training_meta()["num_of_obs"]


@pytest.mark.parametrize(
    "level_knot_dates",
    [
        pd.date_range(start="2016-03-01", end="2019-01-01", freq="3M"),
        pd.date_range(start="2016-03-01", end="2019-01-01", freq="6M"),
    ],
)
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "single", "with_coef": False})], indirect=True
)
def test_ktrlite_level_knot_dates(make_daily_data, level_knot_dates):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        level_knot_dates=level_knot_dates,
        estimator="stan-map",
        n_bootstrap_draws=1e4,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert np.all(np.isin(ktrlite._model.level_knot_dates, level_knot_dates))
    assert len(ktrlite._model.level_knot_dates) == len(level_knot_dates)


@pytest.mark.parametrize("level_knot_distance", [90, 120])
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "single", "with_coef": False})], indirect=True
)
def test_ktrlite_level_knot_distance(make_daily_data, level_knot_distance):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        level_knot_distance=level_knot_distance,
        estimator="stan-map",
        n_bootstrap_draws=1e4,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "seas_segments",
    [
        pytest.param(0, id="0-seas_segement"),
        pytest.param(1, id="1-seas_segement"),
        pytest.param(5, id="5-seas_segement"),
    ],
)
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "single", "with_coef": False})], indirect=True
)
def test_ktrlite_seas_segments(make_daily_data, seas_segments):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        level_segments=10,
        seasonality_segments=seas_segments,
        estimator="stan-map",
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ["date", "prediction"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "single", "with_coef": False})], indirect=True
)
def test_ktrlite_predict_decompose(make_daily_data):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator="stan-map",
        n_bootstrap_draws=1e4,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df, decompose=True)

    expected_columns = [
        "date",
        "prediction_5",
        "prediction",
        "prediction_95",
        "trend_5",
        "trend",
        "trend_95",
        "seasonality_7_5",
        "seasonality_7",
        "seasonality_7_95",
        "seasonality_365.25_5",
        "seasonality_365.25",
        "seasonality_365.25_95",
    ]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "single", "with_coef": False})], indirect=True
)
def test_ktrlite_predict_decompose_point_estimate(make_daily_data):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator="stan-map",
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df, decompose=True)

    expected_columns = [
        "date",
        "prediction",
        "trend",
        "seasonality_7",
        "seasonality_365.25",
    ]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters


def test_ktrlite_hourly_data(ca_hourly_electricity_data):
    train_df = ca_hourly_electricity_data

    ktrlite = KTRLite(
        response_col="SDGE",
        date_col="Dates",
        seasonality=[24, 7, 365.25],
        seasonality_fs_order=[3, 3, 5],
        estimator="stan-map",
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(train_df)

    expected_columns = ["Dates", "prediction"]
    expected_shape = (train_df.shape[0], len(expected_columns))
    expected_num_parameters = len(ktrlite._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    smape_val = smape(train_df["SDGE"].values, predict_df["prediction"].values)
    assert smape_val <= SMAPE_TOLERANCE
