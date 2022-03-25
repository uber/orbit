import pytest
import numpy as np
import pandas as pd
from orbit.utils.knots import get_knot_idx, get_knot_dates


@pytest.mark.parametrize("num_of_segments", [0, 1, 3, 10])
@pytest.mark.parametrize(
    "dataset, date_col",
    [
        ("iclaims_training_data", "week"),
        ("m3_monthly_data", "date"),
        ("ca_hourly_electricity_data", "Dates"),
    ],
)
def test_segments_args(dataset, date_col, num_of_segments, request):
    df = request.getfixturevalue(dataset)
    date_array = df[date_col]
    knot_idx = get_knot_idx(num_of_obs=df.shape[0], num_of_segments=num_of_segments)
    assert knot_idx[0] == 0
    assert len(knot_idx) == num_of_segments + 1
    freq = pd.infer_freq(date_array)
    knot_dates = get_knot_dates(date_array[0], knot_idx, freq)
    # first knot always at the first date
    assert knot_dates[0] == date_array[0]


@pytest.mark.parametrize("knot_distance", [2, 4])
@pytest.mark.parametrize(
    "dataset, date_col",
    [
        ("iclaims_training_data", "week"),
        ("m3_monthly_data", "date"),
        ("ca_hourly_electricity_data", "Dates"),
    ],
)
def test_distance_args(dataset, date_col, knot_distance, request):
    df = request.getfixturevalue(dataset)
    date_array = df[date_col]
    knot_idx = get_knot_idx(num_of_obs=df.shape[0], knot_distance=knot_distance)

    assert knot_idx[0] == 0
    assert knot_idx[3] - knot_idx[2] == knot_distance


@pytest.mark.parametrize(
    "dataset, date_col, knot_dates, knot_idx",
    [
        (
            "iclaims_training_data",
            "week",
            pd.to_datetime(["2014-05-18", "2016-10-30"]),
            np.array([228, 356]),
        ),
        (
            "m3_monthly_data",
            "date",
            pd.to_datetime(["1990-03-01", "1991-03-01"]),
            np.array([2, 14]),
        ),
        (
            "ca_hourly_electricity_data",
            "Dates",
            pd.to_datetime(["2018-01-01 02:00:00", "2018-01-05 05:00:00"]),
            np.array([2, 101]),
        ),
    ],
)
def test_dates_args(dataset, date_col, knot_dates, knot_idx, request):
    df = request.getfixturevalue(dataset)
    date_array = df[date_col]
    knot_idx2 = get_knot_idx(date_array=date_array, knot_dates=knot_dates)
    freq = pd.infer_freq(date_array)
    knot_dates2 = get_knot_dates(date_array[0], knot_idx, freq)

    assert np.all(knot_idx2 == knot_idx)
    assert np.all(knot_dates2 == knot_dates)
