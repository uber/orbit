import pytest
import numpy as np

from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP
from orbit.models.lm import LinearModelFull, LinearModelMAP
from orbit.constants import lm as constants


@pytest.mark.parametrize("model_class", [LinearModelFull, LinearModelMAP])
def test_lm_init(synthetic_data, model_class):
    train_df, test_df, coef = synthetic_data

    lm = model_class(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:]
    )

    is_fitted = lm.is_fitted()

    model_data_input = lm.get_model_data_input()
    model_param_names = lm.get_model_param_names()

    # model is not yet fitted
    assert not is_fitted
    # should only be initialized and not set
    assert not model_data_input
    # model param names should already be set
    assert model_param_names


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC])
def test_lm_full(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lm = LinearModelFull(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        prediction_percentiles=[5, 95],
        num_warmup=50,
        verbose=False,
        estimator_type=estimator_type
    )

    lm.fit(train_df)

    predict_df = lm.predict(test_df)

    expected_columns = ['week', 'prediction_5', 'prediction', 'prediction_95']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len([param.value for param in constants.StanSampleOutput]) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lm._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMAP])
def test_lm_map(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lm = LinearModelMAP(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        prediction_percentiles=[10, 90],
        verbose=False,
        estimator_type=estimator_type
    )

    lm.fit(train_df)
    predict_df = lm.predict(test_df)

    expected_columns = ['week', 'prediction_10', 'prediction', 'prediction_90']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len([param.value for param in constants.StanSampleOutput])

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lm._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 50, 95], [5, 10, 90, 95]])
def test_lm_full_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    lm = LinearModelFull(
        response_col='claims',
        date_col='week',
        regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
        num_warmup=40,
        seed=777,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels = ['_5', '', '_95']
    else:
        if 50 not in prediction_percentiles:
            prediction_percentiles += [50]
        prediction_percentiles = np.sort(prediction_percentiles)
        p_labels = [("_" + str(p) if p != 50 else '')
                    for p in prediction_percentiles]

    lm.fit(df)
    predicted_df = lm.predict(df)
    expected_columns = ['week'] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 50, 95], [5, 10, 90, 95]])
def test_lm_map_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    lm = LinearModelMAP(
        response_col='claims',
        date_col='week',
        regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
        num_warmup=40,
        seed=777,
        prediction_percentiles=prediction_percentiles,
    )

    if not prediction_percentiles:
        p_labels = ['_5', '', '_95']
    else:
        if 50 not in prediction_percentiles:
            prediction_percentiles += [50]
        prediction_percentiles = np.sort(prediction_percentiles)
        p_labels = [("_" + str(p) if p != 50 else '')
                    for p in prediction_percentiles]

    lm.fit(df)
    predicted_df = lm.predict(df)
    expected_columns = ['week'] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]
