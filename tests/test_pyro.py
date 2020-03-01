# import os
# import sys

# import numpy as np
# import pandas as pd
# import pytest
#
# from orbit.lgt import LGT
#
# REPO = os.path.dirname(os.path.dirname(__file__))
# DATA_FILE = os.path.join(REPO, "examples", "data", "iclaims.example.csv")
#
#
# @pytest.mark.parametrize('predict_method', ['map', 'full', 'mean', 'median'])
# def test_smoke(predict_method):
#     raw_df = pd.read_csv(DATA_FILE)
#     raw_df['week'] = pd.to_datetime(raw_df['week'])
#     df = raw_df.copy()
#     df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']] = \
#         df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']].apply(np.log, axis=1)
#
#     test_size = 52
#     train_df = df[:-test_size]
#     test_df = df[-test_size:]
#     lgt_map = LGT(response_col='claims', date_col='week', seasonality=52,
#                   seed=8888,
#                   predict_method=predict_method,
#                   inference_engine='pyro')
#     lgt_map.fit(df=train_df)

from enum import Enum
import pytest
from orbit.lgt import LGT
from orbit.exceptions import IllegalArgument


def test_lgt_pyro_fit(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        inference_engine='pyro'
    )

    lgt.fit(df=iclaims_training_data)
    expected_posterior_parameters = 13

    assert len(lgt.posterior_samples) == expected_posterior_parameters


def test_lgt_pyro_fit_with_missing_input(iclaims_training_data):
    class MockInputMapper(Enum):
        SOME_STAN_INPUT = 'some_stan_input'

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        inference_engine='pyro'
    )

    lgt._stan_input_mapper = MockInputMapper

    with pytest.raises(IllegalArgument):
        lgt.fit(df=iclaims_training_data)


def test_lgt_pyro_fit_and_mean_predict(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        inference_engine='pyro'
    )

    lgt.fit(df=iclaims_training_data)

    predicted_df = lgt.predict(df=iclaims_training_data)

    expected_shape = (443, 2)
    expected_columns = ['week', 'prediction']

    assert predicted_df.shape == expected_shape
    assert list(predicted_df.columns) == expected_columns


def test_lgt_pyro_fit_and_full_predict(iclaims_training_data):
    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        prediction_percentiles=[5, 95, 30],
        predict_method='full',
        sample_method='vi',
        inference_engine='pyro'
    )
    
    lgt.fit(df=iclaims_training_data)

    predicted_out = lgt.predict(df=iclaims_training_data)

    expected_columns = ['week', 5, 30, 50, 95]
    expected_shape = (443, len(expected_columns))

    assert predicted_out.shape == expected_shape
    assert list(predicted_out.columns) == expected_columns
