import pytest
import numpy as np
import pandas as pd

from orbit.models import KTRLite
from orbit.models import KTRX
from orbit.diagnostics.metrics import smape


@pytest.mark.parametrize("regressor_col", [None, ['a', 'b', 'c']])
def test_ktrx_pyro(make_daily_data, regressor_col):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator='stan-map',
    )
    ktrlite.fit(train_df)

    level_knot_dates = ktrlite._model._level_knot_dates
    point_posteriors = ktrlite.get_point_posteriors()
    level_knots = np.squeeze(point_posteriors['lev_knot'], 0)
    sea_coef_knot = point_posteriors['coef_knot']
    # sea_coef_knot = np.squeeze(point_posteriors['coef_knot'], 0)
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktrlite._model._coef_knot_dates,
        '_sea_coef_knot': sea_coef_knot,
        '_seasonality': ktrlite._model._seasonality,
        '_seasonality_fs_order': ktrlite._model._seasonality_fs_order,
    }
    ktrx = KTRX(
        response_col='response',
        date_col='date',
        regressor_col=regressor_col,
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator='pyro-svi',
        n_bootstrap_draws=1e4,
    )

    ktrx.fit(train_df, point_method='median')
    predict_df = ktrx.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= 0.5


@pytest.mark.parametrize("coefficients_knot_dates", [pd.date_range(start='2016-03-01', end='2019-01-01', freq='3M')])
def test_ktrx_coef_knot_dates(make_daily_data, coefficients_knot_dates):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator='stan-map'
    )
    ktrlite.fit(train_df)

    level_knot_dates = ktrlite._model._level_knot_dates
    point_posteriors = ktrlite.get_point_posteriors()
    level_knots = np.squeeze(point_posteriors['lev_knot'], 0)
    sea_coef_knot = point_posteriors['coef_knot']
    # sea_coef_knot = np.squeeze(point_posteriors['coef_knot'], 0)
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktrlite._model._coef_knot_dates,
        '_sea_coef_knot': sea_coef_knot,
        '_seasonality': ktrlite._model._seasonality,
        '_seasonality_fs_order': ktrlite._model._seasonality_fs_order,
    }
    ktrx = KTRX(
        response_col='response',
        date_col='date',
        regressor_col=['a', 'b', 'c'],
        coefficients_knot_dates=coefficients_knot_dates,
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator='pyro-svi',
        n_bootstrap_draws=1e4,
    )

    ktrx.fit(train_df, point_method='median')
    predict_df = ktrx.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrx._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= 0.5


@pytest.mark.parametrize("coefficients_knot_length", [90, 120])
def test_ktrx_coef_knot_distance(make_daily_data, coefficients_knot_length):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLite(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator='stan-map'
    )
    ktrlite.fit(train_df)

    level_knot_dates = ktrlite._model._level_knot_dates
    point_posteriors = ktrlite.get_point_posteriors()
    level_knots = np.squeeze(point_posteriors['lev_knot'], 0)
    sea_coef_knot = point_posteriors['coef_knot']
    # sea_coef_knot = np.squeeze(point_posteriors['coef_knot'], 0)
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktrlite._model._coef_knot_dates,
        '_sea_coef_knot': sea_coef_knot,
        '_seasonality': ktrlite._model._seasonality,
        '_seasonality_fs_order': ktrlite._model._seasonality_fs_order,
    }
    ktrx = KTRX(
        response_col='response',
        date_col='date',
        regressor_col=['a', 'b', 'c'],
        coefficients_knot_length=coefficients_knot_length,
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator='pyro-svi',
        n_bootstrap_draws=1e4,
    )

    ktrx.fit(train_df, point_method='median')
    predict_df = ktrx.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrx._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= 0.5


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

    ktrlite = KTRLite(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator='stan-map'
    )
    ktrlite.fit(train_df)

    level_knot_dates = ktrlite._model._level_knot_dates
    point_posteriors = ktrlite.get_point_posteriors()
    level_knots = np.squeeze(point_posteriors['lev_knot'], 0)
    sea_coef_knot = point_posteriors['coef_knot']
    # sea_coef_knot = np.squeeze(point_posteriors['coef_knot'], 0)
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktrlite._model._coef_knot_dates,
        '_sea_coef_knot': sea_coef_knot,
        '_seasonality': ktrlite._model._seasonality,
        '_seasonality_fs_order': ktrlite._model._seasonality_fs_order,
    }
    ktrx = KTRX(
        response_col='response',
        date_col='date',
        regressor_col=['a', 'b', 'c'],
        regressor_sign=regressor_signs,
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator='pyro-svi',
        n_bootstrap_draws=1e4,
    )

    ktrx.fit(train_df, point_method='median')
    predict_df = ktrx.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrx._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= 0.5


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

    ktrlite = KTRLite(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator='stan-map'
    )
    ktrlite.fit(train_df)

    level_knot_dates = ktrlite._model._level_knot_dates
    point_posteriors = ktrlite.get_point_posteriors()
    level_knots = np.squeeze(point_posteriors['lev_knot'], 0)
    sea_coef_knot = point_posteriors['coef_knot']
    # sea_coef_knot = np.squeeze(point_posteriors['coef_knot'], 0)
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktrlite._model._coef_knot_dates,
        '_sea_coef_knot': sea_coef_knot,
        '_seasonality': ktrlite._model._seasonality,
        '_seasonality_fs_order': ktrlite._model._seasonality_fs_order,
    }
    ktrx = KTRX(
        response_col='response',
        date_col='date',
        regressor_col=['a', 'b', 'c'],
        coef_prior_list=coef_prior_list,
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator='pyro-svi',
        n_bootstrap_draws=1e4,
    )

    ktrx.fit(train_df, point_method='median')
    predict_df = ktrx.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 7

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrx._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= 0.5
