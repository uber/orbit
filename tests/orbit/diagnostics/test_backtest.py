import pytest
import pandas as pd
import numpy as np

from orbit.diagnostics.backtest import TimeSeriesSplitter, BackTester
from orbit.diagnostics.metrics import smape, wmape, mape, mse, mae, rmsse
from orbit.models.lgt import LGTMAP


def test_time_series_splitter():
    np_array = np.random.randn(100, 4)
    df = pd.DataFrame(np_array)

    tss = TimeSeriesSplitter(
        df=df,
        min_train_len=50,
        incremental_len=10,
        forecast_len=10
    )

    expected_number_of_splits = 5

    assert expected_number_of_splits == len(list(tss.split()))

@pytest.mark.parametrize(
    "scheduler_args",
    [
        {
            'min_train_len': 100,
            'incremental_len': 100,
            'forecast_len': 20
        },
        {
            'incremental_len': 100,
            'forecast_len': 20,
            'n_splits': 3
        },
        {
            'forecast_len': 20,
            'n_splits': 1
        },
        {
            'forecast_len': 1,
            'n_splits': 3
        }
    ],
    ids=['use_min_train_len', 'use_n_splits', 'single_split', 'one_forecast_len']
)
def test_backtester_sceduler_args(iclaims_training_data, scheduler_args):
    df = iclaims_training_data

    lgt = LGTMAP(
        response_col='claims',
        date_col='week',
        seasonality=1,
        verbose=False
    )

    backtester = BackTester(
        model=lgt,
        df=df,
        **scheduler_args
    )

    backtester.fit_predict()
    eval_out = backtester.score(metrics=[smape])
    evaluated_metrics = set(eval_out['metric_name'].tolist())

    if metrics is None:
        expected_metrics = [x.__name__ for x in backtester._default_metrics]
    elif isinstance(metrics, list):
        expected_metrics = [x.__name__ for x in metrics]
    else:
        expected_metrics = [metrics.__name__]

    assert set(expected_metrics) == evaluated_metrics

@pytest.mark.parametrize(
    "scheduler_args",
    [
        {
            'min_train_len': 100,
            'incremental_len': 100,
            'forecast_len': 20
        },
        {
            'incremental_len': 100,
            'forecast_len': 20,
            'n_splits': 3
        },
        {
            'forecast_len': 20,
            'n_splits': 1
        }
    ]
)
@pytest.mark.parametrize("metrics", [None, [smape, mape], smape])
def test_backtester_test_data_only(iclaims_training_data, scheduler_args, metrics):
    df = iclaims_training_data

    lgt = LGTMAP(
        response_col='claims',
        date_col='week',
        seasonality=1,
        verbose=False
    )

    backtester = BackTester(
        model=lgt,
        df=df,
        **scheduler_args
    )

    backtester.fit_predict()
    eval_out = backtester.score(metrics=metrics)
    evaluated_metrics = set(eval_out['metric_name'].tolist())

    if metrics is None:
        expected_metrics = [x.__name__ for x in backtester._default_metrics]
    elif isinstance(metrics, list):
        expected_metrics = [x.__name__ for x in metrics]
    else:
        expected_metrics = [metrics.__name__]

    assert set(expected_metrics) == evaluated_metrics


def test_backtester_with_training_data(iclaims_training_data):
    df = iclaims_training_data

    lgt = LGTMAP(
        response_col='claims',
        date_col='week',
        seasonality=1,
        verbose=False
    )

    backtester = BackTester(
        model=lgt,
        df=df,
        min_train_len=100,
        incremental_len=100,
        forecast_len=20,
    )

    backtester.fit_predict()
    eval_out = backtester.score(include_training_metrics=True)
    evaluated_test_metrics = set(eval_out.loc[~eval_out['is_training_metric'], 'metric_name'].tolist())
    evaluated_train_metrics = set(eval_out.loc[eval_out['is_training_metric'], 'metric_name'].tolist())

    expected_test_metrics = [x.__name__ for x in backtester._default_metrics]

    expected_train_metrics = list(filter(
        lambda x: backtester._get_metric_callable_signature(x) == {'actual', 'predicted'}, backtester._default_metrics)
    )
    expected_train_metrics = [x.__name__ for x in expected_train_metrics]

    assert set(expected_test_metrics) == evaluated_test_metrics
    assert set(expected_train_metrics) == evaluated_train_metrics
