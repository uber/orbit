import numpy as np
import pandas as pd


def get_gap_between_dates(start_date, end_date, freq):
    diff = end_date - start_date
    gap = np.array(diff / np.timedelta64(1, freq)).astype(int)

    return gap


def get_knot_idx(num_of_obs, knot_distance):
    """function to calculate the knot idx based on num_of_obs and knot_distance."""
    # starts with the the ending point
    # negative values are allowed when 0 is not included in generated sequence
    knot_idx = np.sort(np.arange(num_of_obs - 1, -1, -knot_distance))
    if 0 not in knot_idx:
        knot_idx = np.sort(np.arange(num_of_obs - 1, -1 - knot_distance, -knot_distance))

    return knot_idx


def get_knot_locations(training_meta, df,
                       knot_dates=None, knot_distance=None, num_of_segments=None, date_freq=None):
    """ function to get the knot locations. This function will be used in KTRLite and KTRX model.
    Parameters
    ----------
    training_meta : dict
        contains the training meta data such as number of observations, date column, training start date, etc
    df : training data frame
    knot_dates : list
        list of dates, which will be used as the knot locations
    knot_distance : int
        distance between every two knots
    num_of_segments : int
        number of segments, which will be used to calculate the knot distance
    date_freq : str
        the date frequency of the input training data

    Returns
    -------
    an array of integers, which are the knot location indices (starts at 0).

    """
    num_of_obs = training_meta['num_of_observations']
    date_col = training_meta['date_col']
    training_start = training_meta['training_start']
    training_end = training_meta['training_end']

    if knot_dates is not None:
        # filtering out knot_dates outside the training data
        _knot_dates = pd.to_datetime([
            x for x in knot_dates if
            (x <= df[date_col].values[-1]) and (x >= df[date_col].values[0])
        ])

        if date_freq is None:
            # infer date freq if not supplied
            date_freq = pd.infer_freq(df[date_col])[0]

        knot_idx = get_gap_between_dates(training_start, _knot_dates, date_freq)

    elif knot_distance is not None:
        if not isinstance(knot_distance, int):
            raise Exception("knot_distance must be an int.")
        knot_idx = get_knot_idx(num_of_obs, knot_distance)
    elif num_of_segments is not None:
        knot_distance = np.round(num_of_obs / num_of_segments).astype(int)
        knot_idx = get_knot_idx(num_of_obs, knot_distance)
    else:
        raise Exception("please specify at least one of the followings to determine the knot locations: "
                        "knot_dates, knot_distance, or num_of_segments.")

    # knot_idx starts with 0; need to add 1 when calculate the fraction
    return knot_idx

