import numpy as np
import pandas as pd


def get_gap_between_dates(start_date, end_date, freq):
    diff = end_date - start_date
    gap = np.array(diff / np.timedelta64(1, freq)).astype(int)

    return gap


def get_knot_locations(training_meta, df,
                       knot_dates=None, knot_distance=None, num_of_segments=None, date_freq=None):
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
        # end points are used, including the first point and the last one
        if not isinstance(knot_distance, int):
            raise Exception("knot_distance must be an int.")
        knot_idx = np.arange(0, num_of_obs, knot_distance)
        if num_of_obs - 1 not in knot_idx:
            knot_idx = np.append(knot_idx, num_of_obs - 1)
    elif num_of_segments is not None:
        # end points are used, including the first point and the last one
        knot_distance = np.round(num_of_obs / num_of_segments).astype(int)
        knot_idx = np.arange(0, num_of_obs, knot_distance)
        if num_of_obs - 1 not in knot_idx:
            knot_idx = np.append(knot_idx, num_of_obs - 1)
    else:
        raise Exception("please specify at least one of the followings to determine the knot locations: "
                        "knot_dates, knot_distance, or num_of_segments.")

    # knot_idx starts with 0; need to add 1 when calculate the fraction
    return knot_idx

