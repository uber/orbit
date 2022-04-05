import pytest
import numpy as np
import pandas as pd

from orbit.models import KTR
from orbit.diagnostics.metrics import smape

SMAPE_TOLERANCE = 0.2


@pytest.mark.parametrize("make_daily_data", [({"seasonality": None})], indirect=True)
def test_ktr_basic(make_daily_data):
    train_df, _, _ = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        estimator="pyro-svi",
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(train_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (train_df.shape[0], len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert np.all(np.isfinite(predict_df["prediction"].values))


@pytest.mark.parametrize(
    "seas_segments",
    [
        pytest.param(0, id="0-seas_segement"),
        pytest.param(1, id="2-seas_segement"),
        pytest.param(5, id="5-seas_segement"),
    ],
)
@pytest.mark.parametrize(
    "make_daily_data, seasonality",
    [
        pytest.param(({"seasonality": "single"}), [365.25], id="single_seasonality"),
        pytest.param(({"seasonality": "dual"}), [7, 365.25], id="dual_seasonality"),
    ],
    indirect=["make_daily_data"],
)
def test_ktr_seasonality(make_daily_data, seasonality, seas_segments):
    train_df, _, _ = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        estimator="pyro-svi",
        seasonality=seasonality,
        seasonality_segments=seas_segments,
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(train_df, decompose=True)
    expected_columns = [
        "date",
        "prediction_5",
        "prediction",
        "prediction_95",
        "trend_5",
        "trend",
        "trend_95",
        "regression_5",
        "regression",
        "regression_95",
    ]

    if len(seasonality) == 1:
        expected_columns += [
            "seasonality_365.25_5",
            "seasonality_365.25",
            "seasonality_365.25_95",
        ]
    if len(seasonality) == 2:
        expected_columns += [
            "seasonality_7_5",
            "seasonality_7",
            "seasonality_7_95",
            "seasonality_365.25_5",
            "seasonality_365.25",
            "seasonality_365.25_95",
        ]
    expected_shape = (train_df.shape[0], len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert np.all(np.isfinite(predict_df["prediction"].values))
    smape_val = smape(train_df["response"].values, predict_df["prediction"].values)
    assert smape_val <= SMAPE_TOLERANCE


@pytest.mark.parametrize("regressor_col", [None, ["a", "b", "c"]])
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "dual", "with_coef": True})], indirect=True
)
def test_ktr_regression(make_daily_data, regressor_col):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=regressor_col,
        estimator="pyro-svi",
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert np.all(np.isfinite(predict_df["prediction"].values))


@pytest.mark.parametrize(
    "regression_knot_dates",
    [pd.date_range(start="2016-03-01", end="2019-01-01", freq="3M")],
)
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "dual", "with_coef": True})], indirect=True
)
def test_ktrx_coef_knot_dates(make_daily_data, regression_knot_dates):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=["a", "b", "c"],
        regression_knot_dates=regression_knot_dates,
        estimator="pyro-svi",
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr.get_posterior_samples()) == expected_num_parameters
    assert np.all(np.isfinite(predict_df["prediction"].values))


@pytest.mark.parametrize("regression_knot_distance", [90, 120])
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "dual", "with_coef": True})], indirect=True
)
def test_ktrx_coef_knot_distance(make_daily_data, regression_knot_distance):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=["a", "b", "c"],
        regression_knot_distance=regression_knot_distance,
        estimator="pyro-svi",
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "regressor_signs",
    [["+", "+", "+"], ["-", "-", "-"], ["=", "=", "="], ["+", "=", "-"]],
    ids=["positive_only", "negative_only", "regular_only", "mixed_signs"],
)
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "dual", "with_coef": True})], indirect=True
)
def test_ktrx_regressor_sign(make_daily_data, regressor_signs):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        regressor_col=["a", "b", "c"],
        regressor_sign=regressor_signs,
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator="pyro-svi",
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters

    coef_df = ktr.get_regression_coefs()
    for i, sign in enumerate(regressor_signs):
        if sign == "+":
            assert all(np.sign(coef_df.iloc[:, 1 + i]) == 1)
        elif sign == "-":
            assert all(np.sign(coef_df.iloc[:, 1 + i]) == -1)


@pytest.mark.parametrize(
    "coef_prior_list",
    [
        [
            {
                "name": "test1",
                "prior_start_tp_idx": 100,
                "prior_end_tp_idx": 120,
                "prior_mean": [0.25, 0.35],
                "prior_sd": [0.1, 0.2],
                "prior_regressor_col": ["a", "b"],
            }
        ]
    ],
)
@pytest.mark.parametrize(
    "make_daily_data", [({"seasonality": "dual", "with_coef": True})], indirect=True
)
def test_ktrx_prior_ingestion(make_daily_data, coef_prior_list):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col="response",
        date_col="date",
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=["a", "b", "c"],
        coef_prior_list=coef_prior_list,
        estimator="pyro-svi",
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ["date", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "regression_segments",
    [
        pytest.param(0, id="0-regression_segments"),
        pytest.param(1, id="1-regression_segments"),
        pytest.param(5, id="5-regression_segments"),
    ],
)
@pytest.mark.parametrize(
    "regressor_col",
    [
        pytest.param(["trend.unemploy"], id="single-regressor"),
        pytest.param(
            ["trend.unemploy", "trend.filling", "trend.job"], id="multiple-regressor"
        ),
    ],
)
def test_ktr_regressor_sizes(iclaims_training_data, regression_segments, regressor_col):
    # this iclaims dataset is raw; the one under load_data is log-transformed
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])

    ktr = KTR(
        date_col="week",
        response_col="claims",
        regressor_col=regressor_col,
        seasonality=[52],
        seasonality_fs_order=[3],
        level_knot_scale=0.1,
        level_segments=10,
        regression_segments=regression_segments,
        regression_rho=0.15,
        # pyro optimization parameters
        seed=8888,
        num_steps=100,
        num_sample=100,
    )
    ktr.fit(df)
    ktr.predict(df)
    coef_df = ktr.get_regression_coefs()
    knot_df = ktr.get_regression_coef_knots()

    expected_columns_coef = ["week"] + regressor_col
    expected_columns_knot = ["week", "step"] + regressor_col

    assert coef_df.columns.tolist() == expected_columns_coef
    assert knot_df.columns.tolist() == expected_columns_knot
    assert coef_df.shape[0] == df.shape[0]
    assert knot_df.shape[0] == regression_segments + 1
