import os
import sys

import numpy as np
import pandas as pd
import pytest

from orbit.lgt import LGT

REPO = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(REPO, "examples", "data", "iclaims.example.csv")


@pytest.mark.parametrize('predict_method', ['map', 'svi'])
def test_smoke(predict_method):
    raw_df = pd.read_csv(DATA_FILE)
    raw_df['week'] = pd.to_datetime(raw_df['week'])
    df = raw_df.copy()
    df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']] = \
        df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']].apply(np.log, axis=1)

    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    lgt_map = LGT(response_col='claims', date_col='week', seasonality=52,
                  seed=8888,
                  predict_method=predict_method,
                  inference_engine='pyro')
    lgt_map.fit(df=train_df)
