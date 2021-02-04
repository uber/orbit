import pytest
import numpy as np

from orbit.estimators.pyro_estimator import PyroEstimator, PyroEstimatorVI, PyroEstimatorMAP
from orbit.estimators.stan_estimator import StanEstimator, StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.ktrlite import BaseKTRLite, KTRLiteFull, KTRLiteMAP
from orbit.constants.constants import PredictedComponents

@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP, PyroEstimatorMAP])
def test_ktrlite_map(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktr = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=365.25,
        seasonality_fs_order=2,
        estimator_type=estimator_type
    )

    ktr.fit(train_df)
    predict_df = ktr.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_ktrlite_dual_seasonality(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktr = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        estimator_type=StanEstimatorMAP
    )

    ktr.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_ktrlite_level_knot_dates(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktr = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        level_knot_dates=pd.to_datime(['2016-01-01', '2017-01-01', '2018-01-01'])
        estimator_type=StanEstimatorMAP
    )

    ktr.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_ktrlite_level_knot_distance(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktr = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        level_knot_dates=pd.to_datime(['2016-01-01', '2017-01-01', '2018-01-01'])
        estimator_type=StanEstimatorMAP
    )

    ktr.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_ktrlite_coef_knot_distance(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktr = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=[7, 365.25],
        seasonality_fs_order=[2, 5],
        level_knot_dates=pd.to_datime(['2016-01-01', '2017-01-01', '2018-01-01'])
        estimator_type=StanEstimatorMAP
    )

    ktr.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['day', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 8

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters



@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP, PyroEstimatorMAP])
def test_lgt_aggregated_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    ktr = KTRLiteMAP(
        response_col='response',
        date_col='day',
        seasonality=365.25,
        seasonality_fs_order=5,
        estimator_type=estimator_type
    )

    ktr.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    ktr = KTRLiteMAP(
        response_col='claims',
        date_col='week',
        seasonality=52,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels  = ['_5', '', '_95']
    else:
        p_labels = ['_5', '_10', '', '_95']

    lgt.fit(df)
    predicted_df = lgt.predict(df)
    expected_columns = ['week'] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    predicted_df = lgt.predict(df, decompose=True)
    plot_components = [
        'prediction',
        PredictedComponents.TREND.value,
        PredictedComponents.SEASONALITY.value,
        PredictedComponents.REGRESSION.value]

    expected_columns = ['week']
    for pc in plot_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]



