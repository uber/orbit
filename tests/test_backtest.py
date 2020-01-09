import numpy as np
import pandas as pd
import pytest

from orbit.exceptions import BacktestException
from orbit.utils.constants import BacktestFitColumnNames

from orbit.lgt import LGT
from orbit.backtest import BacktestEngine


def test_backtest_meta(iclaims_training_data):
    lgt = LGT(
            response_col='claims',
            date_col='week',
            seasonality=52,
            chains=4,
            predict_method='map',
        )
    bt = BacktestEngine(lgt, iclaims_training_data)
    min_train_len = 300
    forecast_len = 20
    incremental_len = 20
    bt.create_meta(min_train_len, incremental_len, forecast_len, scheme='expanding')
    expected_bt_iterations = 7

    assert len(bt.bt_meta) == expected_bt_iterations


def test_backtest_run(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='map'
    )

    bt = BacktestEngine(lgt, iclaims_training_data)
    min_train_len = 300
    forecast_len = 20
    incremental_len = 20
    bt.create_meta(min_train_len, incremental_len, forecast_len, scheme='expanding')
    bt.run(verbose=False, save_results=False, pred_col='prediction')

    expected_shape = (140, 6)
    expected_columns = [col.value for col in BacktestFitColumnNames]

    assert bt.bt_res.shape == expected_shape
    assert list(bt.bt_res.columns) == expected_columns


def test_backtest_invalid_scheme(iclaims_training_data):
    with pytest.raises(BacktestException):
        lgt = LGT(
            response_col='claims',
            date_col='week',
            seasonality=52,
            chains=4,
            predict_method='map',
        )
        bt = BacktestEngine(lgt, iclaims_training_data)
        min_train_len = 300
        forecast_len = 20
        incremental_len = 20
        bt.create_meta(min_train_len, incremental_len, forecast_len, scheme='unknown')
        return bt

