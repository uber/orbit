from enum import Enum
import numpy as np
import pandas as pd
import pytest
from orbit.lgt import LGT
from orbit.exceptions import IllegalArgument, EstimatorException

from orbit.constants.constants import COEFFICIENT_DF_COLS


@pytest.fixture(scope='module')
def lgt_mcmc_mean(synthetic_data):
    train_df, test_df, coef = synthetic_data
    lgt = LGT(
        response_col='response',
        date_col='week',
        seasonality=52,
        sample_method='mcmc',
        predict_method='mean'
    )
    lgt.fit(train_df)
    return lgt


@pytest.fixture(scope='module')
def lgt_mcmc_full(synthetic_data):
    train_df, test_df, coef = synthetic_data
    lgt = LGT(
        response_col='response',
        date_col='week',
        seasonality=52,
        sample_method='mcmc',
        predict_method='full'
    )
    lgt.fit(train_df)
    return lgt


@pytest.fixture(scope='module')
def lgt_mcmc_mean_with_regression(synthetic_data):
    train_df, test_df, coef = synthetic_data
    lgt = LGT(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        seasonality=52,
        sample_method='mcmc',
        predict_method='mean',
    )
    lgt.fit(train_df)
    return lgt


@pytest.mark.parametrize("sample_method", ["map", "vi", "mcmc"])
@pytest.mark.parametrize("predict_method", ["map", "mean", "median", "full"])
def test_fit_sample_and_predict_args(synthetic_data, sample_method, predict_method):
    train_df, test_df, coef = synthetic_data

    valid_permutations = [
        ("map", "map"),
        ("vi", "mean"), ("vi", "median"), ("vi", "full"),
        ("mcmc", "mean"), ("mcmc", "median"), ("mcmc", "full")
    ]

    if (sample_method, predict_method) in valid_permutations:
        lgt = LGT(
            response_col='response',
            date_col='week',
            seasonality=52,
            sample_method=sample_method,
            predict_method=predict_method,
            num_warmup=400
        )

        lgt.fit(train_df)
        predict_df = lgt.predict(test_df)

        # assert number of posterior param keys
        if sample_method == 'map':
            assert len(lgt.posterior_samples) == 23
        else:
            assert len(lgt.posterior_samples) == 13

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
                sample_method=sample_method,
                predict_method=predict_method,
                num_warmup=400
            )
            lgt.fit(train_df)


@pytest.mark.parametrize("sample_method", ["map", "vi", "mcmc"])
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
def test_fit_with_regression(synthetic_data, sample_method, predict_method, regressor_signs):
    train_df, test_df, coef = synthetic_data

    valid_permutations = [
        ("map", "map"),
        ("vi", "mean"), ("vi", "median"), ("vi", "full"),
        ("mcmc", "mean"), ("mcmc", "median"), ("mcmc", "full")
    ]

    if (sample_method, predict_method) in valid_permutations:
        lgt = LGT(
            response_col='response',
            date_col='week',
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            sample_method=sample_method,
            predict_method=predict_method,
            num_warmup=400
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
                sample_method=sample_method,
                predict_method=predict_method,
                num_warmup=400
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

    lgt._stan_input_mapper = MockInputMapper

    with pytest.raises(EstimatorException):
        lgt.fit(df=iclaims_training_data)


def test_lgt_invalid_init_params():
    with pytest.raises(IllegalArgument):
        lgt = LGT(
            some_non_existent_param='invalid'
        )
        return lgt


def test_lgt_predict_decompose(iclaims_training_data):

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean'
    )

    lgt.fit(df=iclaims_training_data)

    predicted_out = lgt.predict(df=iclaims_training_data, decompose=True)

    expected_shape = (443, 5)
    expected_columns = ['week', 'prediction', 'trend', 'seasonality', 'regression']

    print(predicted_out.head())

    assert predicted_out.shape == expected_shape
    assert list(predicted_out.columns) == expected_columns


def test_negative_lgt_predict_mcmc(iclaims_training_data):

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='full',
        sample_method='mcmc'

    )

    lgt.fit(df=iclaims_training_data)

    with pytest.raises(IllegalArgument):
        lgt.predict(df=iclaims_training_data, decompose=True)


def test_lgt_forecast(iclaims_training_data):

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean'
    )

    lgt.fit(df=iclaims_training_data)

    forecast_df = iclaims_training_data.copy()
    forecast_df['week'] = forecast_df['week'] + np.timedelta64(30, 'W')

    predicted_out = lgt.predict(df=iclaims_training_data, decompose=True)
    predicted_out_forecast = lgt.predict(df=forecast_df, decompose=True)

    predicted_out_filtered = predicted_out.iloc[30:]['trend'].reset_index(drop=True)
    predicted_out_forecast_filtered = predicted_out_forecast.iloc[:-30]['trend']\
        .reset_index(drop=True)

    assert predicted_out_forecast.shape == (443, 5)

    # trend term should be the same during the overlapping in-sample period
    assert predicted_out_filtered.equals(predicted_out_forecast_filtered)

    # TODO implement negative case with forecast


def test_lgt_with_regressors(iclaims_training_data):

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        regressor_col=['trend.unemploy', 'trend.filling']
    )

    lgt.fit(df=iclaims_training_data)

    predicted_df = lgt.predict(df=iclaims_training_data, decompose=False)

    expected_shape = (443, 2)
    expected_columns = ['week', 'prediction']

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_lgt_with_regressors_negative(iclaims_training_data):

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


def test_lgt_with_regressors_and_forecast(iclaims_training_data):

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        prediction_percentiles=[5, 95, 30],
        predict_method='full',
        sample_method='mcmc',
        regressor_col=['trend.unemploy', 'trend.filling']
    )

    lgt.fit(df=iclaims_training_data)

    forecast_df = iclaims_training_data.copy()
    forecast_df['week'] = forecast_df['week'] + np.timedelta64(100, 'W')

    # predicted_dict = lgt._vectorized_predict(df=forecast_df, include_error=True)

    predicted_df = lgt.predict(df=forecast_df, decompose=False)

    expected_columns = ['week', 5, 30, 50, 95]
    expected_shape = (443, len(expected_columns))

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_get_regression_coefs(iclaims_training_data):
    regressor_cols = ['trend.unemploy', 'trend.filling', 'trend.job']

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        prediction_percentiles=[5, 95, 30],
        predict_method='full',
        sample_method='mcmc',
        regressor_col=regressor_cols,
        regressor_sign=["=", "=", "+"]
    )

    lgt.fit(df=iclaims_training_data)

    reg_coefs = lgt.get_regression_coefs()

    assert set(reg_coefs[COEFFICIENT_DF_COLS.REGRESSOR]) == set(regressor_cols)

    # negative case
    with pytest.raises(IllegalArgument):
        lgt.get_regression_coefs(aggregation_method='full')


def test_lgt_multiple_fits(m3_monthly_data):

    lgt = LGT(response_col='value',
              date_col='date',
              seasonality=12,
              sample_method='mcmc',
              predict_method='full')

    # multiple fits should not raise exceptions
    lgt.fit(df=m3_monthly_data)
    lgt.fit(df=m3_monthly_data)

    predicted_df = lgt.predict(df=m3_monthly_data)

    expected_columns = ['date', 5, 50, 95]
    expected_shape = (68, len(expected_columns))

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns
