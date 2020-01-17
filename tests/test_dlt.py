import numpy as np
import pandas as pd
import pytest
from orbit.dlt import DLT
from orbit.exceptions import IllegalArgument


def test_dlt_fit(iclaims_training_data):
    dlt = DLT(
            response_col='claims',
            date_col='week',
            seasonality=52,
            chains=4,
        )

    dlt.get_params()
    dlt.fit(df=iclaims_training_data)

    expected_posterior_parameters = 13

    assert len(dlt.posterior_samples) == expected_posterior_parameters


def test_dlt_fit_and_mean_predict(iclaims_training_data):
    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean'
    )

    dlt.fit(df=iclaims_training_data)

    predicted_df = dlt.predict(df=iclaims_training_data)

    expected_shape = (443, 2)
    expected_columns = ['week', 'prediction']

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_dlt_fit_and_mcmc_predict(iclaims_training_data):
    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        prediction_percentiles=[5, 95, 30],
        predict_method='full',
        sample_method='mcmc'
    )

    dlt.fit(df=iclaims_training_data)

    predicted_out = dlt.predict(df=iclaims_training_data)

    expected_columns = ['week', 5, 30, 50, 95]
    expected_shape = (443, len(expected_columns))

    assert predicted_out.shape == expected_shape
    assert list(predicted_out.columns) == expected_columns


def test_dlt_invalid_init_params():
    with pytest.raises(IllegalArgument):
        dlt = DLT(
            some_non_existent_param='invalid'
        )
        return dlt


def test_dlt_predict_decompose(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean'
    )

    dlt.fit(df=iclaims_training_data)

    predicted_out = dlt.predict(df=iclaims_training_data, decompose=True)

    expected_shape = (443, 5)
    expected_columns = ['week', 'prediction', 'trend', 'seasonality', 'regression']

    print(predicted_out.head())

    assert predicted_out.shape == expected_shape
    assert list(predicted_out.columns) == expected_columns


def test_negative_dlt_predict_mcmc(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='full',
        sample_method='mcmc'

    )

    dlt.fit(df=iclaims_training_data)

    with pytest.raises(IllegalArgument):
        dlt.predict(df=iclaims_training_data, decompose=True)


def test_dlt_forecast(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean'
    )

    dlt.fit(df=iclaims_training_data)

    forecast_df = iclaims_training_data.copy()
    forecast_df['week'] = forecast_df['week'] + np.timedelta64(30, 'W')

    predicted_out = dlt.predict(df=iclaims_training_data, decompose=True)
    predicted_out_forecast = dlt.predict(df=forecast_df, decompose=True)

    predicted_out_filtered = predicted_out.iloc[30:]['trend'].reset_index(drop=True)
    predicted_out_forecast_filtered = predicted_out_forecast.iloc[:-30]['trend']\
        .reset_index(drop=True)

    assert predicted_out_forecast.shape == (443, 5)

    # trend term should be the same during the overlapping period
    assert predicted_out_filtered.equals(predicted_out_forecast_filtered)

    # TODO implement negative case with forecast


def test_dlt_with_regressors(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        regressor_col=['trend.unemploy', 'trend.filling']
    )

    dlt.fit(df=iclaims_training_data)

    predicted_df = dlt.predict(df=iclaims_training_data, decompose=False)

    expected_shape = (443, 2)
    expected_columns = ['week', 'prediction']

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_dlt_with_regressors_negative(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        regressor_col=['wrong_column_name']
    )

    with pytest.raises(IllegalArgument):
        dlt.fit(df=iclaims_training_data)


def test_predict_subset_of_train(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
    )

    dlt.fit(df=iclaims_training_data)

    predicted_df = dlt.predict(df=iclaims_training_data[:100])

    expected_shape = (100, 4)
    expected_start_date = pd.to_datetime('2010-01-03')
    expected_end_date = pd.to_datetime('2011-11-27')

    assert predicted_df.shape == expected_shape
    assert min(predicted_df['week']) == expected_start_date
    assert max(predicted_df['week']) == expected_end_date


def test_invalid_date_order():

    dlt = DLT(
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
        dlt.fit(df1)

    # catch unordered weeks
    with pytest.raises(IllegalArgument):
        dlt.fit(df2)


def test_dlt_with_regressors_and_forecast(iclaims_training_data):

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        prediction_percentiles=[5, 95, 30],
        predict_method='full',
        sample_method='mcmc',
        regressor_col=['trend.unemploy', 'trend.filling']
    )

    dlt.fit(df=iclaims_training_data)

    forecast_df = iclaims_training_data.copy()
    forecast_df['week'] = forecast_df['week'] + np.timedelta64(100, 'W')

    # predicted_dict = dlt._vectorized_predict(df=forecast_df, include_error=True)

    predicted_df = dlt.predict(df=forecast_df, decompose=False)

    expected_columns = ['week', 5, 30, 50, 95]
    expected_shape = (443, len(expected_columns))

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns
