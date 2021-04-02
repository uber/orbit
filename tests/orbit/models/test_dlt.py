import pytest
import numpy as np
from copy import copy

from orbit.models.dlt import DLTFull, DLTAggregated, DLTMAP
from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI


@pytest.mark.parametrize("model_class", [DLTMAP, DLTFull, DLTAggregated])
def test_base_ets_init(model_class):
    dlt = model_class()

    is_fitted = dlt.is_fitted()

    model_data_input = dlt._get_model_data_input()
    model_param_names = dlt._get_model_param_names()
    init_values = dlt._get_init_values()

    # model is not yet fitted
    assert not is_fitted
    # should only be initialized and not set
    assert not model_data_input
    # model param names should already be set
    assert model_param_names
    # callable is not implemented yet
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

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
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

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
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

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])

    predict_df = dlt.predict(test_df, decompose=True)
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

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
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

    dlt = DLTMAP(
        response_col='claims',
        date_col='week',
        regressor_col=raw_regressor_col,
        regressor_sign=regressor_signs,
        seasonality=52,
        seed=8888,
    )
    dlt.fit(df)
    predicted_df_v1 = dlt.predict(df)
    predicted_df_v2 = dlt.predict(new_df)

    # mixing ordering of signs
    dlt_new = DLTMAP(
        response_col='claims',
        date_col='week',
        regressor_col=new_regressor_col,
        regressor_sign=new_regressor_signs,
        seasonality=52,
        seed=8888,
    )
    dlt_new.fit(df)
    predicted_df_v3 = dlt_new.predict(df)
    predicted_df_v4 = dlt_new.predict(new_df)

    pred_v1 = predicted_df_v1['prediction'].values
    pred_v2 = predicted_df_v2['prediction'].values
    pred_v3 = predicted_df_v3['prediction'].values
    pred_v4 = predicted_df_v4['prediction'].values

    # they should be all identical; ordering of signs or columns in prediction show not matter
    assert np.allclose(pred_v1, pred_v2, atol=1e-3)
    assert np.allclose(pred_v1, pred_v3, atol=1e-3)
    assert np.allclose(pred_v1, pred_v4, atol=1e-3)


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["-", "-", "-", "-", "-", "-"],
        ["+", "=", "+", "=", "+", "+"],
        ["-", "=", "-", "=", "-", "="],
        ["+", "=", "+", "=", "-", "-"],
    ],
    ids=['positive_only', 'regular_only', 'negative_only',
         'positive_mixed', 'negative_mixed', 'mixed_signs']
)
@pytest.mark.parametrize("seasonality", [1, 52])
def test_dlt_full_reproducibility(synthetic_data, estimator_type, regressor_signs, seasonality):
    train_df, test_df, coef = synthetic_data

    dlt_first = DLTFull(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    # first fit and predict
    dlt_first.fit(train_df)
    posteriors_first = copy(dlt_first._posterior_samples)
    predict_df_first = dlt_first.predict(test_df)
    regression_out_first = dlt_first.get_regression_coefs()

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    dlt_second = DLTFull(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    dlt_second.fit(train_df)
    posteriors_second = copy(dlt_second._posterior_samples)
    predict_df_second = dlt_second.predict(test_df)
    regression_out_second = dlt_second.get_regression_coefs()

    # assert same posterior keys
    assert set(posteriors_first.keys()) == set(posteriors_second.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors_first.items():
        assert np.allclose(posteriors_first[k], posteriors_second[k])

    # assert identical regression columns
    # this is also checked in posterior samples, but an extra layer just in case
    # since this one very commonly retrieved by end users
    assert regression_out_first.equals(regression_out_second)

    # assert prediction is reproducible
    assert predict_df_first.equals(predict_df_second)


@pytest.mark.parametrize("seasonality", [1, 52])
def test_dlt_map_reproducibility(synthetic_data, seasonality):
    train_df, test_df, coef = synthetic_data

    dlt1 = DLTMAP(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
    )

    # first fit and predict
    dlt1.fit(train_df)
    posteriors1 = copy(dlt1._aggregate_posteriors['map'])
    prediction1 = dlt1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    dlt2 = DLTMAP(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
    )

    dlt2.fit(train_df)
    posteriors2 = copy(dlt2._aggregate_posteriors['map'])
    prediction2 = dlt2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert np.allclose(prediction1['prediction'].values, prediction2['prediction'].values)


@pytest.mark.parametrize("regression_penalty", ['fixed_ridge', 'lasso', 'auto_ridge'])
def test_dlt_regression_penalty(synthetic_data, regression_penalty):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        regression_penalty=regression_penalty,
        seasonality=52,
        num_warmup=50,
        verbose=False,
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


@pytest.mark.parametrize("level_sm_input", [0.0001, 0.5, 1.0])
@pytest.mark.parametrize("seasonality_sm_input", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("slope_sm_input", [0.0, 0.5, 1.0])
def test_dlt_fixed_sm_input(synthetic_data, level_sm_input, seasonality_sm_input, slope_sm_input):
    train_df, test_df, coef = synthetic_data

    dlt = DLTMAP(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        level_sm_input=level_sm_input,
        seasonality_sm_input=seasonality_sm_input,
        slope_sm_input=slope_sm_input,
        seasonality=52,
        num_warmup=50,
        verbose=False,
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