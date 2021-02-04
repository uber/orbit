import pytest
import numpy as np

from orbit.estimators.pyro_estimator import PyroEstimator, PyroEstimatorVI, PyroEstimatorMAP
from orbit.estimators.stan_estimator import StanEstimator, StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.ktrx import BaseKTRX, KTRXFull, KTRXAggregated, KTRXMAP
from orbit.constants.constants import PredictedComponents

@pytest.mark.parametrize("estimator_type", [PyroEstimatorVI])
def test_ktrx_pyro(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=365.25,
        seasonality_fs_order=2,
        estimator_type=StanEstimatorMAP
    )

    level_knot_dates = ktr_stan._level_knot_dates
    level_knots = ktr_stan._aggregate_posteriors['map']['lev_knot'][0]
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktr_stan._coef_knot_dates,
        '_sea_coef_knot': ktr_stan._aggregate_posteriors['map']['coef_knot'],
        '_sea_rho': ktr_stan.rho_coefficients,
        '_seasonality': ktr_stan._seasonality,
        '_seasonality_fs_order': ktr_stan._seasonality_fs_order,
    }
    ktrx = KTRXAggregated(
        response_col='response',
        date_col='day',
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator_type=estimator_type,
    )

    ktrx.fit(train_df)
    predict_df = ktrx.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [PyroEstimatorVI])
def test_ktrx_coef_knot_distance(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=365.25,
        seasonality_fs_order=2,
        estimator_type=StanEstimatorMAP
    )

    level_knot_dates = ktr_stan._level_knot_dates
    level_knots = ktr_stan._aggregate_posteriors['map']['lev_knot'][0]
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktr_stan._coef_knot_dates,
        '_sea_coef_knot': ktr_stan._aggregate_posteriors['map']['coef_knot'],
        '_sea_rho': ktr_stan.rho_coefficients,
        '_seasonality': ktr_stan._seasonality,
        '_seasonality_fs_order': ktr_stan._seasonality_fs_order,
    }
    ktrx = KTRXAggregated(
        response_col='response',
        date_col='day',
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        estimator_type=estimator_type,
    )

    ktrx.fit(train_df)
    predict_df = ktrx.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_ktrx_pyro(synthetic_data, regressor_signs):
    train_df, test_df, coef = synthetic_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=365.25,
        seasonality_fs_order=2,
        estimator_type=StanEstimatorMAP
    )

    level_knot_dates = ktr_stan._level_knot_dates
    level_knots = ktr_stan._aggregate_posteriors['map']['lev_knot'][0]
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktr_stan._coef_knot_dates,
        '_sea_coef_knot': ktr_stan._aggregate_posteriors['map']['coef_knot'],
        '_sea_rho': ktr_stan.rho_coefficients,
        '_seasonality': ktr_stan._seasonality,
        '_seasonality_fs_order': ktr_stan._seasonality_fs_order,
    }
    ktrx = KTRXAggregated(
        response_col='response',
        date_col='day',
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        estimator_type=estimator_type,
    )

    ktrx.fit(train_df)
    predict_df = ktrx.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "coef_prior_list",
    [
        [
            {
                'name': 'test1',
                'prior_start_tp_idx': 100,
                'prior_end_tp_idx': 200,
                'prior_mean': [0, 0],
                'prior_sd': [1, 1],
                'prior_regressor_col': ['reg2', 'reg3']

            }
        ]
    ]
)
def test_ktrx_pyro(synthetic_data, coef_prior_list):
    train_df, test_df, coef = synthetic_data

    ktrlite = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=365.25,
        seasonality_fs_order=2,
        estimator_type=StanEstimatorMAP
    )

    level_knot_dates = ktr_stan._level_knot_dates
    level_knots = ktr_stan._aggregate_posteriors['map']['lev_knot'][0]
    seasonal_knots_input = {
        '_seas_coef_knot_dates': ktr_stan._coef_knot_dates,
        '_sea_coef_knot': ktr_stan._aggregate_posteriors['map']['coef_knot'],
        '_sea_rho': ktr_stan.rho_coefficients,
        '_seasonality': ktr_stan._seasonality,
        '_seasonality_fs_order': ktr_stan._seasonality_fs_order,
    }
    ktrx = KTRXAggregated(
        response_col='response',
        date_col='day',
        level_knot_dates=level_knot_dates,
        level_knots=level_knots,
        seasonal_knots_input=seasonal_knots_input,
        regressor_col=train_df.columns.tolist()[2:],
        coef_prior_list=coef_prior_list,
        regressor_sign=regressor_signs,
        estimator_type=estimator_type,
    )

    ktrx.fit(train_df)
    predict_df = ktrx.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters