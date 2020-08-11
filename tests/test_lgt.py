from enum import Enum
import numpy as np
import pandas as pd
import pytest
from orbit.lgt import LGT
from orbit.exceptions import IllegalArgument, EstimatorException

from orbit_v1.models.lgt import BaseLGT, LGTFull, LGTAggregated
from orbit_v1.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI


@pytest.mark.parametrize("infer_method", ["map", "vi", "mcmc"])
@pytest.mark.parametrize("predict_method", ["map", "mean", "median", "full"])
def test_fit_and_predict_univariate(
        synthetic_data, infer_method, predict_method, valid_sample_predict_method_combo):
    train_df, test_df, coef = synthetic_data

    if (infer_method, predict_method) in valid_sample_predict_method_combo:
        lgt = LGT(
            response_col='response',
            date_col='week',
            seasonality=52,
            infer_method=infer_method,
            predict_method=predict_method,
            num_warmup=50,
        )

        lgt.fit(train_df)
        predict_df = lgt.predict(test_df)

        # assert number of posterior param keys
        if predict_method == 'full':
            assert len(lgt.posterior_samples) == 13
        else:
            assert len(lgt.aggregated_posteriors[predict_method]) == 12

        # assert output shape
        if predict_method == 'full':
            expected_columns = ['week', 5, 50, 95]
            expected_shape = (51, len(expected_columns))
            assert predict_df.shape == expected_shape
            assert predict_df.columns.tolist() == expected_columns
        else:
            expected_columns = ['week', 'prediction']
            expected_shape = (51, len(expected_columns))
            assert predict_df.shape == expected_shape
            assert predict_df.columns.tolist() == expected_columns

    else:
        with pytest.raises(EstimatorException):
            lgt = LGT(
                response_col='response',
                date_col='week',
                seasonality=52,
                infer_method=infer_method,
                predict_method=predict_method,
                num_warmup=50
            )
            lgt.fit(train_df)


@pytest.mark.parametrize("seasonality_sm_input", [-1, 0.5])
def test_non_seasonal_fit(m3_monthly_data, seasonality_sm_input):
    mod = LGT(response_col='value',
              date_col='date',
              seasonality=-1,
              seasonality_sm_input=seasonality_sm_input,
              infer_method='map',
              predict_method='map')

    mod.fit(df=m3_monthly_data)
    predicted_df = mod.predict(df=m3_monthly_data)
    assert len(mod.aggregated_posteriors['map']) == 10
    assert predicted_df.shape == (68, 2)

@pytest.mark.parametrize("infer_method", ["map", "vi", "mcmc"])
@pytest.mark.parametrize("predict_method", ["map", "mean", "median", "full"])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"]
    ],
    ids=['positive_only', 'regular_only', 'mixed_signs']
)
def test_fit_and_predict_with_regression(
        synthetic_data, infer_method, predict_method,
        regressor_signs, valid_sample_predict_method_combo):
    train_df, test_df, coef = synthetic_data

    if (infer_method, predict_method) in valid_sample_predict_method_combo:
        lgt = LGT(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            infer_method=infer_method,
            predict_method=predict_method,
            num_warmup=50
        )

        lgt.fit(train_df)
        predict_df = lgt.predict(test_df)

        num_regressors = lgt.get_regression_coefs().shape[0]

        assert num_regressors == len(train_df.columns.tolist()[2:])

        # assert output shape
        if predict_method == 'full':
            expected_columns = ['week', 5, 50, 95]
            expected_shape = (51, len(expected_columns))

            assert predict_df.shape == expected_shape
            assert predict_df.columns.tolist() == expected_columns
        else:
            expected_columns = ['week', 'prediction']
            expected_shape = (51, len(expected_columns))

            assert predict_df.shape == expected_shape
            assert predict_df.columns.tolist() == expected_columns

    else:
        with pytest.raises(EstimatorException):
            lgt = LGT(
                response_col='response',
                date_col='week',
                seasonality=52,
                infer_method=infer_method,
                predict_method=predict_method,
                num_warmup=50
            )
            lgt.fit(train_df)


@pytest.mark.parametrize("infer_method", ["map", "vi", "mcmc"])
@pytest.mark.parametrize("predict_method", ["map", "mean", "median", "full"])
def test_fit_and_decomp_with_regression(
        synthetic_data, infer_method, predict_method, valid_sample_predict_method_combo):
    train_df, test_df, coef = synthetic_data

    if (infer_method, predict_method) in valid_sample_predict_method_combo:
        lgt = LGT(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            seasonality=52,
            infer_method=infer_method,
            predict_method=predict_method,
            num_warmup=50
        )
        lgt.fit(train_df)

        # full should raise illegal argument
        if predict_method == 'full':
            with pytest.raises(IllegalArgument):
                lgt.predict(test_df, decompose=True)

        else:
            predict_df = lgt.predict(test_df, decompose=True)

            expected_columns = ['week', 'prediction', 'trend', 'seasonality', 'regression']
            expected_shape = (51, len(expected_columns))

            assert predict_df.shape == expected_shape
            assert predict_df.columns.tolist() == expected_columns

    else:
        with pytest.raises(EstimatorException):
            lgt = LGT(
                response_col='response',
                date_col='week',
                seasonality=52,
                infer_method=infer_method,
                predict_method=predict_method,
                num_warmup=50
            )
            lgt.fit(train_df)


def test_lgt_fit_with_missing_input(iclaims_training_data):
    class MockInputMapper(Enum):
        SOME_STAN_INPUT = 'some_stan_input'

    lgt = LGT(
            response_col='claims',
            date_col='week',
            seasonality=52,
            chains=4,
        )

    lgt._data_input_mapper = MockInputMapper

    with pytest.raises(EstimatorException):
        lgt.fit(df=iclaims_training_data)


def test_lgt_invalid_init_params():
    with pytest.raises(IllegalArgument):
        lgt = LGT(
            some_non_existent_param='invalid'
        )
        return lgt


def test_invalid_regressor_column(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        regressor_col=['wrong_column_name']
    )

    with pytest.raises(IllegalArgument):
        lgt.fit(df=iclaims_training_data)


def test_predict_subset_of_train(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
    )

    lgt.fit(df=iclaims_training_data)

    predicted_df = lgt.predict(df=iclaims_training_data[:100])

    expected_shape = (100, 4)
    expected_start_date = pd.to_datetime('2010-01-03')
    expected_end_date = pd.to_datetime('2011-11-27')

    assert predicted_df.shape == expected_shape
    assert min(predicted_df['week']) == expected_start_date
    assert max(predicted_df['week']) == expected_end_date


def test_invalid_date_order():
    lgt = LGT(
        response_col='claims',
        date_col='week',
    )

    claims = np.random.randn(5)
    week1 = pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-03', '2019-01-04'])
    week2 = pd.to_datetime(['2019-01-01', '2019-01-02', '2019-01-03', '2018-12-31', '2019-01-04'])

    df1 = pd.DataFrame({'week': week1, 'claims': claims})
    df2 = pd.DataFrame({'week': week2, 'claims': claims})

    # catch repeating weeks
    with pytest.raises(IllegalArgument):
        lgt.fit(df1)

    # catch unordered weeks
    with pytest.raises(IllegalArgument):
        lgt.fit(df2)


def test_fit_monthly_data(m3_monthly_data):
    lgt = LGT(response_col='value',
              date_col='date',
              seasonality=12,
              infer_method='mcmc',
              predict_method='full')

    # multiple fits should not raise exceptions
    lgt.fit(df=m3_monthly_data)
    lgt.fit(df=m3_monthly_data)

    predicted_df = lgt.predict(df=m3_monthly_data)

    expected_columns = ['date', 5, 50, 95]
    expected_shape = (68, len(expected_columns))

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_fit_and_predict_with_regression_all_int(synthetic_data):
    train_df, test_df, coef = synthetic_data
    reg_columns = train_df.columns.tolist()[2:]

    # convert to all int
    train_df[reg_columns] = train_df[reg_columns].astype(np.int) + 1
    test_df[reg_columns] = test_df[reg_columns].astype(np.int) + 1

    lgt = LGT(
        response_col='response',
        date_col='week',
        regressor_col=reg_columns,
        seasonality=52,
        infer_method='mcmc',
        predict_method='mean',
        num_warmup=50,
        is_multiplicative=False
    )
    lgt.fit(train_df)

    predict_df = lgt.predict(df=test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns


def test_base_lgt_init():
    lgt = BaseLGT()

    is_fitted = lgt.is_fitted()

    stan_data_input = lgt._get_stan_data_input()
    model_param_names = lgt._get_model_param_names()
    stan_init = lgt._get_stan_init()

    assert not is_fitted  # model is not yet fitted
    assert not stan_data_input  # should only be initialized and not set
    assert model_param_names  # model param names should already be set
    # todo: change when stan_init callable is implemented
    assert not stan_init


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

    expected_columns = ['week', 5, 'prediction', 95]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 13

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator_type", [StanEstimatorMCMC, StanEstimatorVI])
def test_lgt_non_seasonal_fit(synthetic_data, estimator_type):
    train_df, test_df, coef = synthetic_data

    lgt = LGTFull(
        response_col='response',
        date_col='week',
        estimator_type=estimator_type
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    expected_columns = ['week', 'prediction']
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = 11

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
