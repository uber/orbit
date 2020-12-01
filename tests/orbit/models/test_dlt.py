import pytest
import numpy as np

from orbit.models.dlt import BaseDLT, DLTFull, DLTAggregated, DLTMAP
from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI


def test_base_dlt_init():
    dlt = BaseDLT()

    is_fitted = dlt.is_fitted()

    model_data_input = dlt._get_model_data_input()
    model_param_names = dlt._get_model_param_names()
    init_values = dlt._get_init_values()

    assert not is_fitted  # model is not yet fitted
    assert not model_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when init_values callable is implemented
    assert not init_values


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_dlt_full_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    dlt = DLTFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_dlt_aggregated_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    dlt = DLTAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


def test_dlt_map_univariate(synthetic_data):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` parameter in optimizing()

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_dlt_non_seasonal_fit(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    dlt = DLTFull(
        response_col='response',
        date_col='week',
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 11

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["-", "-", "-", "-", "-", "-"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "-", "-"]
    ],
    ids=['positive_only', 'negative_only', 'regular_only', 'mixed_signs']
)
def test_dlt_full_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    dlt = DLTFull(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["-", "-", "-", "-", "-", "-"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "-", "-"]
    ],
    ids=['positive_only', 'negative_only', 'regular_only', 'mixed_signs']
)
def test_dlt_aggregated_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    dlt = DLTAggregated(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])
    assert any(predict_df['regression'].values)


@pytest.mark.parametrize("global_trend_option", ["linear", "loglinear", "logistic", "flat"])
def test_dlt_map_global_trend(synthetic_data, global_trend_option):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        global_trend_option=global_trend_option,
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    "regressor_signs",
    [
        ['=', '=', '+'],
        ['=', '=', '-'],
        ['+', '-', '+'],
        ['+', '-', '=']
    ],
    ids=['positive_mixed', 'negative_mixed', 'positive_negative', 'mixed']
)
def test_dlt_mixed_signs_and_order(iclaims_training_data, regressor_signs):
    df = iclaims_training_data
    df['claims'] = np.log(df['claims'])
    raw_regressor_col = ['trend.unemploy', 'trend.filling', 'trend.job']
    new_regressor_col = [raw_regressor_col[idx] for idx in [1, 2, 0]]
    new_regressor_signs = [regressor_signs[idx] for idx in [1, 2, 0]]
    # mixiing ordering of cols in df of prediction
    new_df = df[['claims', 'week'] + new_regressor_col]

    lgt = DLTMAP(
        response_col='claims',
        date_col='week',
        regressor_col=raw_regressor_col,
        regressor_sign=regressor_signs,
        seasonality=52,
        seed=8888,
    )
    lgt.fit(df)
    predicted_df_v1 = lgt.predict(df)
    predicted_df_v2 = lgt.predict(new_df)

    # mixing ordering of signs
    lgt_new = DLTMAP(
        response_col='claims',
        date_col='week',
        regressor_col=new_regressor_col,
        regressor_sign=new_regressor_signs,
        seasonality=52,
        seed=8888,
    )
    lgt_new.fit(df)
    predicted_df_v3 = lgt_new.predict(df)
    predicted_df_v4 = lgt_new.predict(new_df)

    pred_v1 = predicted_df_v1['prediction'].values
    pred_v2 = predicted_df_v2['prediction'].values
    pred_v3 = predicted_df_v3['prediction'].values
    pred_v4 = predicted_df_v4['prediction'].values

    # they should be all identical; ordering of signs or columns in prediction show not matter
    assert np.allclose(pred_v1, pred_v2)
    assert np.allclose(pred_v1, pred_v3)
    assert np.allclose(pred_v1, pred_v4)


