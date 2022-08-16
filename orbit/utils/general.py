import os
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product


def update_dict(original_dict, append_dict):
    new_dict = deepcopy(original_dict)
    if append_dict:
        for k, v in append_dict.items():
            new_dict[k] = v
    return new_dict


def is_ordered_datetime(array):
    """Returns True if array is ordered and non-repetitive"""
    return np.all(np.diff(array).astype(float) > 0)


def is_even_gap_datetime(array):
    """Returns True if array is evenly distributed"""
    if len(array) >= 3:
        return isinstance(pd.infer_freq(array), str)
    return True


def is_empty_dataframe(df):
    """A simple function to tell whether the passed in df is an empty dataframe or not.
    Parameters
    ----------
    df : pd.DataFrame
        given input dataframe

    Returns
    -------
    bool : True if df is none, or if df is an empty dataframe; False otherwise.
    """
    return df is None or (isinstance(df, pd.DataFrame) and df.empty)


def get_parent_path(current_file_path):
    """
    Parameters
    ----------
    current_file_path : str
        The given file path, should be an absolute path

    Returns:
    -------
        str : The parent path of give file path
    """

    return os.path.abspath(os.path.join(current_file_path, os.pardir))


def expand_grid(base):
    """Given a base key values span, expand them into a dataframe covering all combinations
    Parameters
    ----------
    base : dict
        dictionary with keys equal columns name and value equals key values

    Returns
    -------
    pd.DataFrame : dataframe generate based on user specified base
    """
    return pd.DataFrame([row for row in product(*base.values())], columns=base.keys())


def regenerate_base_df(df, time_col, key_col, val_cols=[], fill_na=None):
    """Given a dataframe, key column, time column and value column, re-generate multiple time-series to cover full range
    date-time with all the keys.  This can be a useful utils for working multiple time-series.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    key_col : str
    val_cols : List[str]; values column considered to be imputed
    fill_na : Optional[float]; values to fill when there are missing values of the row

    Returns
    -------

    """
    out = df.copy()
    unique_time = out[time_col].unique()
    unique_key = out[key_col].unique()
    new_df_base = expand_grid(
        {
            key_col: unique_key,
            time_col: unique_time,
        }
    )
    out = new_df_base.merge(out, how="left", on=[time_col, key_col])
    if not isinstance(val_cols, list):
        val_cols = list(val_cols)
    out = out[[time_col, key_col] + val_cols]
    if fill_na is not None:
        out[val_cols] = out[val_cols].fillna(fill_na)
    return out
