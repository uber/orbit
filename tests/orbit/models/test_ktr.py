import pytest
import numpy as np
import pandas as pd

from orbit.models import KTRLite, KTR
from orbit.diagnostics.metrics import smape

SMAPE_TOLERANCE = 0.6


@pytest.mark.parametrize("regressor_col", [None, ['a', 'b', 'c']])
def test_ktrx_pyro(make_daily_data, regressor_col):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=regressor_col,
        estimator='pyro-svi',
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize("regression_knot_dates", [pd.date_range(start='2016-03-01', end='2019-01-01', freq='3M')])
def test_ktrx_coef_knot_dates(make_daily_data, regression_knot_dates):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=['a', 'b', 'c'],
        regression_knot_dates=regression_knot_dates,
        estimator='pyro-svi',
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize("regression_knot_distance", [90, 120])
def test_ktrx_coef_knot_distance(make_daily_data, regression_knot_distance):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=['a', 'b', 'c'],
        regression_knot_distance=regression_knot_distance,
        estimator='pyro-svi',
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+"],
        ["=", "=", "="],
        ["+", "=", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_ktrx_regressor_sign(make_daily_data, regressor_signs):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col='response',
        date_col='date',
        regressor_col=['a', 'b', 'c'],
        regressor_sign=regressor_signs,
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator='pyro-svi',
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize(
    "coef_prior_list",
    [
        [
            {
                'name': 'test1',
                'prior_start_tp_idx': 100,
                'prior_end_tp_idx': 120,
                'prior_mean': [.25, 0.35],
                'prior_sd': [.1, .2],
                'prior_regressor_col': ['a', 'b']
            }
        ]
    ]
)
def test_ktrx_prior_ingestion(make_daily_data, coef_prior_list):
    train_df, test_df, coef = make_daily_data

    ktr = KTR(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        regressor_col=['a', 'b', 'c'],
        coef_prior_list=coef_prior_list,
        estimator='pyro-svi',
        num_steps=100,
        num_sample=100,
        n_bootstrap_draws=-1,
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktr._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE
