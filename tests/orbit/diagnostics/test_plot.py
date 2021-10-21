import pytest
import pandas as pd
import numpy as np

from orbit.diagnostics.backtest import BackTester
from orbit.diagnostics.metrics import smape
from orbit.models import LGT, DLT, ETS
from orbit.diagnostics.plot import (
    plot_predicted_data,
    plot_predicted_components,
    plot_bt_predictions,
    # plot_posterior_params,
    # plot_param_diagnostics
)


@pytest.mark.parametrize(
    "plot_components", [
        ['trend', 'seasonality', 'regression'],
        ['trend', 'seasonality'],
        ['trend']
    ]
)
def test_plot_predicted_data(iclaims_training_data, plot_components):
    df = iclaims_training_data
    df['claims'] = np.log(df['claims'])

    regressor_col = ['trend.unemploy', 'trend.filling', 'trend.job']
    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    lgt = LGT(
        response_col='claims',
        date_col='week',
        regressor_col=regressor_col,
        estimator='stan-map',
        seasonality=52,
        seed=8888,
    )
    lgt.fit(train_df)
    predicted_df = lgt.predict(df=test_df, decompose=True)

    # test plotting
    _ = plot_predicted_data(training_actual_df=train_df,
                            predicted_df=predicted_df,
                            date_col='week',
                            actual_col='claims',
                            test_actual_df=test_df)

    _ = plot_predicted_components(predicted_df=predicted_df,
                                  date_col='week',
                                  plot_components=plot_components)


# @pytest.mark.parametrize(
#     "plot_kind", ['density', 'trace', 'pair']
# )
# @pytest.mark.parametrize(
#     "plot_which", ['density', 'trace', 'pair', 'forest', 'posterior', 'autocorr']
# )
# def test_plot_predicted_data(iclaims_training_data, plot_kind, plot_which):
#     df = iclaims_training_data
#     df['claims'] = np.log(df['claims'])
#
#     regressor_col = ['trend.unemploy', 'trend.filling', 'trend.job']
#     test_size = 52
#     train_df = df[:-test_size]
#     test_df = df[-test_size:]
#
#     dlt = DLT(response_col='claims',
#               date_col='week',
#               regressor_col=regressor_col,
#               seasonality=52,
#               num_warmup=100,
#               num_sample=100,
#               chains=4)
#     dlt.fit(train_df)
#     predicted_df = dlt.predict(df=test_df, decompose=True)

    # test plotting
    # _ = plot_posterior_params(dlt, kind=plot_kind)
    # _ = plot_param_diagnostics(dlt, which=plot_which)

def test_plot_predicted_data(iclaims_training_data):
    df = iclaims_training_data
    df['claims'] = np.log(df['claims'])

    regressor_col = ['trend.unemploy', 'trend.filling', 'trend.job']

    dlt = DLT(date_col='week',
              response_col='claims',
              regressor_col=regressor_col,
              seasonality=52,
              estimator='stan-map')
    bt = BackTester(model=dlt,
                    df=df,
                    min_train_len=100,
                    incremental_len=100,
                    forecast_len=20)
    bt.fit_predict()
    predicted_df = bt.get_predicted_df()

    # test plotting
    _ = plot_bt_predictions(predicted_df, metrics=smape, ncol=2, include_vline=True)

# TODO: not using in-house plotting anymore can change this to testing arviz integration
# def test_plot_posterior_params(iclaims_training_data):
#     df = iclaims_training_data
#     date_col = 'week'
#     response_col = 'claims'
#     regressor_col = ['trend.unemploy', 'trend.filling', 'trend.job']
#
#     df[response_col] = np.log(df[response_col])
#
#     # without regressor
#     ets = ETS(
#         response_col=response_col,
#         date_col=date_col,
#         seasonality=52,
#         seed=8888,
#     )
#     ets.fit(df=iclaims_training_data)
#     # check get_posterior_samples function
#     posterior_samples = ets.get_posterior_samples()
#     _ = plot_posterior_params(ets, kind='pair', params=['lev_sm', 'sea_sm', 'obs_sigma'], n_bins=10)
#
#     # with regressor
#     dlt = DLT(
#         response_col=response_col,
#         date_col=date_col,
#         regressor_col=regressor_col,
#         seasonality=52,
#         seed=8888,
#     )
#     dlt.fit(df=iclaims_training_data)
#     # check get_posterior_samples function
#     posterior_samples = ets.get_posterior_samples()
#     _ = plot_posterior_params(ets, kind='pair', params=['lev_sm', 'sea_sm', 'obs_sigma'], n_bins=10)
