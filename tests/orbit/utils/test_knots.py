import pytest
import numpy as np
import pandas as pd
from orbit.utils.knots import get_knot_idx, get_knot_dates


@pytest.mark.parametrize(
    "num_of_segments", [0, 1, 3, 10]
)
def test_segments_args(iclaims_training_data, num_of_segments):
    df = iclaims_training_data
    date_array = df['week']
    knot_idx = get_knot_idx(num_of_obs=df.shape[0], num_of_segments=num_of_segments)
    assert knot_idx[0] == 0
    assert len(knot_idx) == num_of_segments + 1
    knot_dates = get_knot_dates(date_array[0], knot_idx, date_array[1] - date_array[0])
    # first knot always at the first date
    assert knot_dates[0] == date_array[0]


@pytest.mark.parametrize(
    "knot_distance", [2, 4]
)
def test_distance_args(iclaims_training_data, knot_distance):
    df = iclaims_training_data
    date_array = df['week']
    knot_idx = get_knot_idx(num_of_obs=df.shape[0], knot_distance=knot_distance)
    assert knot_idx[0] == 0
    assert knot_idx[3] - knot_idx[2] == knot_distance
    knot_dates = get_knot_dates(date_array[0], knot_idx, date_array[1] - date_array[0])
    # first knot always at the first date
    assert knot_dates[0] == date_array[0]
    # the first knot may not always equal knot distance due to appending the first date as the first knot
    assert knot_dates[3] == knot_dates[2] + np.timedelta64(knot_distance, 'W')


@pytest.mark.parametrize(
    "knot_dates", [pd.to_datetime(['2014-05-18', '2016-10-30'])]
)
def test_dates_args(iclaims_training_data, knot_dates):
    df = iclaims_training_data
    date_array = df['week']
    knot_idx = get_knot_idx(date_array=date_array, knot_dates=knot_dates)
    expected_dates = get_knot_dates(date_array[0], knot_idx, date_array[1] - date_array[0])
    expected_idx = np.array([228, 356])

    assert np.all(knot_idx == expected_idx)
    assert np.all(knot_dates == expected_dates)
