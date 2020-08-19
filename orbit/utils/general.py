import os
import numpy as np
import pandas as pd
from copy import deepcopy


def update_dict(original_dict, append_dict):
    new_dict = deepcopy(original_dict)
    if append_dict:
        for k, v in append_dict.items():
            new_dict[k] = v
    return new_dict


def is_ordered_datetime(array):
    """Returns True if array is ordered and non-repetitive"""
    return np.all(np.diff(array).astype(float) > 0)


def is_empty_dataframe(df):
    """
    A simple function to tell whether the passed in df is an empty dataframe or not.
    Parameters
    ----------
    df: pd.DataFrame
        given input dataframe

    Returns
    -------
        boolean
        True if df is none, or if df is an empty dataframe; False otherwise.
    """
    return df is None or (isinstance(df, pd.DataFrame) and df.empty)


def get_parent_path(current_file_path):
    """
    Parameters
    ----------
    current_file_path: str
        The given file path, should be an absolute path

    Returns:
    -------
        str
        The parent path of give file path
    """

    return os.path.abspath(os.path.join(current_file_path, os.pardir))