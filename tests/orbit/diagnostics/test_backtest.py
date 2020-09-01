import pandas as pd
import numpy as np

from orbit.diagnostics.backtest import TimeSeriesSplitter, Backtester
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


def test_backtester(iclaims_training_data):
    df = iclaims_training_data

    lgt = LGTMAP(
        response_col='claims',
        date_col='week',
        seasonality=1,
        verbose=False
    )

    backtester = Backtester(
        model=lgt,
        df=df,
        min_train_len=100,
        incremental_len=100,
        forecast_len=20,
    )

    backtester.fit_score()

    assert True
