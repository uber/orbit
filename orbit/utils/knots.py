import numpy as np
import pandas as pd
from ..exceptions import IllegalArgument


def get_dates_from_idx(start_date, idx, freq):
    """
    Parameters
    ----------
    start_date : numpy datetime
        the starting date to anchor the start of indexing
    idx : ndarray
        1D array containing index with `int` type.
    freq : date frequency

    Returns
    -------
    list :
        list of knot dates with provided start date time and indices
    """
    dates_lst = pd.date_range(start=start_date, periods=max(idx) + 1, freq=freq)
    dates_array = dates_lst[idx]

    return dates_array


def get_idx_from_dates(start_date, date_array, time_delta):
    """return knot index based on date difference normalized with the number of steps by frequency provided

    Parameters
    ----
    start_date : numpy datetime
        the starting date to anchor the start of indexing
    date_array : numpy datetime array
    time_delta : np.timedelta
    """
    date_diff = date_array - start_date
    # time_delta = np.mean(np.diff(date_array))
    # round-up normalized delta can be deemed as indices converted from an array
    date_idx = np.round(date_diff / time_delta).astype(int)
    # to make output consistence
    date_idx = np.array(date_idx)

    return date_idx


def get_knot_idx_by_dist(num_of_steps, knot_distance):
    """function to calculate the knot idx based on num_of_steps and knot_distance."""
    # starts with the the ending point
    # use negative values or simply append 0 to the sequence?
    knot_idx = np.sort(np.arange(num_of_steps - 1, -1, -knot_distance))
    knot_idx = np.round(knot_idx).astype('int')
    if 0 not in knot_idx:
        knot_idx = np.sort(np.append(knot_idx, 0))

    return knot_idx


def get_knot_idx(
        num_of_steps=None,
        num_of_segments=None,
        knot_distance=None,
        date_array=None,
        knot_dates=None,
        time_delta=None,
    ):
    """ function to calculate and return the knot locations as indices based on
    This function will be used in KTRLite and KTRX model.

    There are three ways to get the knot index (in descending priority when all requirements are met):
    1. With observations date array and knot dates provided, derive knots location directly based on the
    implied observation frequency provided.
    2. With number of time steps supplied, calculate the knot location and indices based on
    knot distance specified such that there will be additional knots in the first and end provided
    3. With number of time steps supplied, calculate the knot location and indices based on
    number of segments specified and knot indices will be evenly distributed

    Parameters
    ----------
    num_of_steps : int
        number of steps to derive segments interval and knots location; this will be ignored if knot_dates is not None
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
    if knot_dates is None and num_of_steps is None:
        raise IllegalArgument('Either knot_dates or num_of_steps needs to be provided.')

    if knot_dates is not None:
        if date_array is None:
            raise IllegalArgument('When knot_dates are supplied, users need to supply date_array as well.')

        knot_dates = np.array(knot_dates, dtype='datetime64')

        # filter knots dates outside the training range
        _knot_dates = pd.to_datetime([
            x for x in knot_dates if
            (x <= date_array.max()) and (x >= date_array.min())
        ])

        knot_idx = get_idx_from_dates(
            start_date=date_array[0],
            date_array=_knot_dates,
            time_delta=time_delta,
        )

    elif knot_distance is not None:
        if not isinstance(knot_distance, int):
            raise Exception("knot_distance must be an int.")
        knot_idx = get_knot_idx_by_dist(num_of_steps, knot_distance)

    elif num_of_segments is not None:
        if num_of_segments >= 1:
            knot_distance = (num_of_steps - 1) / num_of_segments
            knot_idx = get_knot_idx_by_dist(num_of_steps, knot_distance)
        else:
            # when the segment is zero, just put a single knot at the beginning
            knot_idx = np.array([0])
    else:
        raise Exception("please specify at least one of the followings to determine the knot locations: "
                        "knot_dates, knot_distance, or num_of_segments.")

    # knot_idx starts with 0; need to add 1 when calculate the fraction
    return knot_idx
