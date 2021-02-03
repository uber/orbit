import copy
import pytest
import numpy as np

from orbit.estimators.pyro_estimator import PyroEstimator, PyroEstimatorVI, PyroEstimatorMAP
from orbit.estimators.stan_estimator import StanEstimator, StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP
from orbit.models.lgt import BaseLGT, LGTFull, LGTAggregated, LGTMAP
from orbit.constants.constants import PredictedComponents


def test_base_lgt_init():
    lgt = BaseLGT()

    is_fitted = lgt.is_fitted()

    model_data_input = lgt._get_model_data_input()
    model_param_names = lgt._get_model_param_names()
    init_values = lgt._get_init_values()

    assert not is_fitted  # model is not yet fitted
    assert not model_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when init_values callable is implemented
    assert not init_values


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_full_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_lgt_full_univariate_pyro(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_steps=10,
        verbose=False,
        estimator_type=PyroEstimatorVI
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` in pyro

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_aggregated_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_lgt_aggregated_univariate_pyro(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTAggregated(
        response_col='response',
        date_col='week',
        seasonality=52,
        verbose=False,
        num_steps=10,
        estimator_type=PyroEstimatorVI
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` in pyro

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP, PyroEstimatorMAP])
def test_lgt_map_univariate(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTMAP(
        response_col='response',
        date_col='week',
        seasonality=52,
        verbose=False,
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 12  # no `lp__` parameter in optimizing()

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_non_seasonal_fit(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        estimator_type=estimator_type,
        num_warmup=50,
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 11

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


def test_lgt_non_seasonal_fit_pyro(synthetic_data):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        estimator_type=PyroEstimatorVI,
        num_steps=10
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 10  # no `lp__` in pyro

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI])
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
def test_lgt_full_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    if issubclass(estimator_type, StanEstimator):
        lgt = LGTFull(
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
    elif issubclass(estimator_type, PyroEstimator):
        lgt = LGTFull(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_steps=10,
            verbose=False,
            estimator_type=estimator_type
        )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI, PyroEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["-", "-", "-", "-", "-", "-"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "-", "-"],
    ],
    ids=['positive_only', 'negative_only', 'regular_only', 'mixed_signs']
)
def test_lgt_aggregated_with_regression(synthetic_data, estimator_type, regressor_signs):
    train_df, test_df, coef = synthetic_data

    if issubclass(estimator_type, StanEstimator):
        lgt = LGTAggregated(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            num_warmup=50,
            verbose=False,
            estimator_type=estimator_type
        )
    elif issubclass(estimator_type, PyroEstimator):
        lgt = LGTAggregated(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            num_steps=10,
            verbose=False,
            estimator_type=estimator_type
        )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])

    predict_df = lgt.predict(test_df, decompose=True)
    assert any(predict_df['regression'].values)


@pytest.mark.parametrize(
    "regressor_signs",
    [
        ['=', '=', '+'],
    ],
    ids=['positive_mixed']
)
def test_lgt_mixed_signs_and_order(iclaims_training_data, regressor_signs):
    df = iclaims_training_data
    df['claims'] = np.log(df['claims'])
    raw_regressor_col = ['trend.unemploy', 'trend.filling', 'trend.job']
    new_regressor_col = [raw_regressor_col[idx] for idx in [2, 1, 0]]
    new_regressor_signs = [regressor_signs[idx] for idx in [2, 1, 0]]
    # mixiing ordering of cols in df of prediction
    new_df = df[['claims', 'week'] + new_regressor_col]

    lgt = LGTMAP(
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
    lgt_new = LGTMAP(
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
    assert np.allclose(pred_v1, pred_v2, atol=1e-3)
    assert np.allclose(pred_v1, pred_v3, atol=1e-3)
    assert np.allclose(pred_v1, pred_v4, atol=1e-3)


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    lgt = LGTFull(
        response_col='claims',
        date_col='week',
        seasonality=52,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels = ['_5', '', '_95']
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


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
@pytest.mark.parametrize("seasonality", [1, 52])
def test_lgt_full_reproducibility(synthetic_data, estimator_type, regressor_signs, seasonality):
    train_df, test_df, coef = synthetic_data

    lgt_first = LGTFull(
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
    lgt_first.fit(train_df)
    posteriors_first = copy.copy(lgt_first._posterior_samples)
    predict_df_first = lgt_first.predict(test_df)
    regression_out_first = lgt_first.get_regression_coefs()

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    lgt_second = LGTFull(
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

    lgt_second.fit(train_df)
    posteriors_second = copy.copy(lgt_second._posterior_samples)
    predict_df_second = lgt_second.predict(test_df)
    regression_out_second = lgt_second.get_regression_coefs()

    posterior_keys = posteriors_first.keys()

    # assert same posterior keys
    assert set(posteriors_first.keys()) == set(posteriors_second.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors_first.items():
        assert np.allclose(posteriors_first[k], posteriors_second[k])

    # assert identical regression columns
    # this is also checked in posterior samples, but an extra layer just in case
    # since this one very commonly retrieved by end users
    assert all(regression_out_first == regression_out_second)

    # assert prediction is reproducible
    assert all(predict_df_first == predict_df_second)


@pytest.mark.parametrize("level_sm_input", [0.0001, 0.5, 1.0])
@pytest.mark.parametrize("seasonality_sm_input", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("slope_sm_input", [0.0, 0.5, 1.0])
def test_lgt_fixed_sm_input(synthetic_data, level_sm_input, seasonality_sm_input, slope_sm_input):
    train_df, test_df, coef = synthetic_data

    lgt = LGTMAP(
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

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])
