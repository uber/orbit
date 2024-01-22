import numpy as np
import pandas as pd
from ..exceptions import IllegalArgument


def get_knot_dates(start_date, knot_idx, freq):
    """
    Parameters
    ----------
    start_date : datetime array
    knot_idx : ndarray
        1D array containing index with `int` type.
    freq : date frequency

    Returns
    -------
    list :
        list of knot dates with provided start date time and indices
    """
    # knot_dates = knot_idx * freq + start_date
    # knot_dates = knot_idx * np.timedelta64(1, freq) + start_date
    dates_lst = pd.date_range(start=start_date, periods=max(knot_idx) + 1, freq=freq)
    knot_dates = dates_lst[knot_idx]

    return knot_dates


def get_dates_delta(start_date, end_date, time_delta):
    """return knot index based on date difference normalized with the number of steps by frequency provided

    Parameters
    ----
    start_date : numpy datetime
    end_date : numpy datetime array
    time_delta : time delta between dates
    """
    date_diff = end_date - start_date
    # can also be deemed as the "knot_idx"
    norm_delta = np.round(date_diff / time_delta).astype(int)

    return norm_delta


def get_knot_idx_by_dist(num_of_obs, knot_distance):
    """function to calculate the knot idx based on num_of_obs and knot_distance."""
    # starts with the the ending point
    # use negative values or simply append 0 to the sequence?
    knot_idx = np.sort(np.arange(num_of_obs - 1, -1, -knot_distance))
    knot_idx = np.round(knot_idx).astype("int")
    if 0 not in knot_idx:
        # knot_idx = np.sort(np.arange(num_of_obs - 1, -1 - knot_distance, -knot_distance))
        knot_idx = np.sort(np.append(knot_idx, 0))

    return knot_idx


def get_knot_idx(
    num_of_obs=None,
    num_of_segments=None,
    knot_distance=None,
    date_array=None,
    knot_dates=None,
):
    """function to calculate and return the knot locations as indices based on
    This function will be used in KTRLite and KTRX model.

    There are three ways to get the knot index:
    1. With number of observations supplied, calculate the knots location and indices based on
    number of segments specified and knot indices will be evenly distributed
    2. With number of observations supplied, calculate the knots location and indices based on
    knot distance specified such that there will be additional knots in the first and end provided
    3. With observations date array and knot dates provided, derive knots location directly based on the
    implied observation frequency provided.

    Parameters
    ----------
    num_of_obs : int
        number of observations to derive segments and knots; will be ignored if knot_dates is not None
    num_of_segments : int
        number of segments, which will be used to calculate the knot distance
    knot_distance : int
        distance between every two knots
    date_array : datetime array
        only used when knot_dates is not None
    knot_dates : list or array of numpy datetime
        list of dates in string format (%Y-%m-%d) or numpy datetime array which will be used as the knot locations

    Returns
    -------
    an array of integers, which are the knot location indices (starts at 0).

    """
    if knot_dates is None and num_of_obs is None:
        raise IllegalArgument("Either knot_dates or num_of_obs needs to be provided.")

    if knot_dates is not None:
        if date_array is None:
            raise IllegalArgument(
                "When knot_dates are supplied, users need to supply date_array as well."
            )
        knot_dates = np.array(knot_dates, dtype="datetime64")

        # filter out
        _knot_dates = pd.to_datetime(
            [
                x
                for x in knot_dates
                if (x <= date_array.max()) and (x >= date_array.min())
            ]
        )

        time_delta = np.diff(date_array).mean()

        knot_idx = get_dates_delta(
            start_date=date_array[0], end_date=_knot_dates, time_delta=time_delta
        )

    elif knot_distance is not None:
        if not isinstance(knot_distance, int):
            raise Exception("knot_distance must be an int.")
        knot_idx = get_knot_idx_by_dist(num_of_obs, knot_distance)

    elif num_of_segments is not None:
        if num_of_segments >= 1:
            knot_distance = (num_of_obs - 1) / num_of_segments
            knot_idx = get_knot_idx_by_dist(num_of_obs, knot_distance)
        else:
            # one single knot at the beginning
            knot_idx = np.array([0])
    else:
        raise Exception(
            "please specify at least one of the followings to determine the knot locations: "
            "knot_dates, knot_distance, or num_of_segments."
        )

    # knot_idx starts with 0; need to add 1 when calculate the fraction
    return knot_idx
