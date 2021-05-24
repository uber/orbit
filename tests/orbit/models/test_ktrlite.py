import pytest
import numpy as np
import pandas as pd

from orbit.estimators.stan_estimator import StanEstimatorMAP
from orbit.models.ktrlite import KTRLiteMAP
from orbit.diagnostics.metrics import smape

SMAPE_TOLERANCE = 0.5


@pytest.mark.parametrize(
    "seasonality_fs_order", [None, [5]],
    ids=['default_order', 'manual_order']
)
def test_ktrlite_single_seas(make_daily_data, seasonality_fs_order):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[365.25],
        seasonality_fs_order=seasonality_fs_order,
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize(
    "seasonality_fs_order", [None, [2, 5]],
    ids=['default_order', 'manual_order']
)
def test_ktrlite_dual_seas(make_daily_data, seasonality_fs_order):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=seasonality_fs_order,
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize("span_level", [.05, .1, .5])
def test_ktrlite_span_level(make_daily_data, span_level):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        span_level=span_level,
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE
    knots_df = ktrlite.get_level_knots()
    levels_df = ktrlite.get_levels()
    assert knots_df.shape[0] == round(1/span_level)
    assert levels_df.shape[0] == ktrlite.num_of_observations


@pytest.mark.parametrize("level_knot_dates", [pd.date_range(start='2016-03-01', end='2019-01-01', freq='3M'),
                                              pd.date_range(start='2016-03-01', end='2019-01-01', freq='6M')])
def test_ktrlite_level_knot_dates(make_daily_data, level_knot_dates):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        level_knot_dates=level_knot_dates,
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE
    assert np.all(np.isin(ktrlite.level_knot_dates, level_knot_dates))
    assert len(ktrlite.level_knot_dates) == len(level_knot_dates)


@pytest.mark.parametrize("level_knot_length", [90, 120])
def test_ktrlite_level_knot_distance(make_daily_data, level_knot_length):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        level_knot_length=level_knot_length,
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


@pytest.mark.parametrize("coefficients_knot_length", [90, 120])
def test_ktrlite_coef_knot_distance(make_daily_data, coefficients_knot_length):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        coefficients_knot_length=coefficients_knot_length,
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


def test_ktrlite_predict_decompose(make_daily_data):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator_type=StanEstimatorMAP
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df, decompose=True)

    expected_columns = ['date', 'prediction_5', 'prediction', 'prediction_95',
                                'trend_5', 'trend', 'trend_95',
                                'seasonality_7_5', 'seasonality_7', 'seasonality_7_95',
                                'seasonality_365.25_5', 'seasonality_365.25', 'seasonality_365.25_95']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE


def test_ktrlite_predict_decompose_point_estimate(make_daily_data):
    train_df, test_df, coef = make_daily_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='date',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator_type=StanEstimatorMAP,
        n_bootstrap_draws=-1,
    )

    ktrlite.fit(train_df)
    predict_df = ktrlite.predict(test_df, decompose=True)

    expected_columns = ['date',  'prediction', 'trend', 'seasonality_7', 'seasonality_365.25']
    expected_shape = (364, len(expected_columns))
    expected_num_parameters = 6

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(ktrlite._posterior_samples) == expected_num_parameters
    assert smape(test_df['response'].values, predict_df['prediction'].values) <= SMAPE_TOLERANCE
