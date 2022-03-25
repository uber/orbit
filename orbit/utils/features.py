import numpy as np
import pandas as pd
from orbit.exceptions import IllegalArgument


def make_fourier_series(n, period, order=3, shift=0):
    """Given time series length, cyclical period and order, return a set of fourier series.

    Parameters
    ----------
    n : int
        Length of time series
    period : int
        Length of a cyclical period. E.g., with daily data, `period = 7` means weekly seasonality.
    order : int
        Number of components for each sin() or cos() series.
    shift : int
        shift of time step/index to generate the series
    Returns
    -------
    2D array-like
        2D array where each column represents the series with a specific order fourier constructed by sin() or cos().
    Notes
    -----
        1. See https://otexts.com/fpp2/complexseasonality.html
        2. Original idea from https://github.com/facebook/prophet under
    """
    t = np.arange(1, n + 1) + shift
    out = list()
    for i in range(1, order + 1):
        x = 2.0 * i * np.pi * t / period
        out.append(np.cos(x))
        out.append(np.sin(x))
    out = np.column_stack(out)
    return out


def make_fourier_series_df(df, period, order=3, prefix="", suffix="", shift=0):
    """Given a data-frame, cyclical period and order, return a set of fourier series in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to supply datetime array for generating fourier series
    period : int
        Length of a cyclical period. E.g., with daily data, `period = 7` means weekly seasonality.
    order : int
        Number of components for each sin() or cos() series.
    prefix : str
        prefix of output columns label
    suffix : str
        suffix of output columns label
    shift : int
        shift of time step/index to generate the series
    Returns
    -------
    out : pd.DataFrame
        data with computed fourier series attached
    fs_cols : list
        list of labels derived from fourier series
    Notes
    -----
        This is calling :func:`make_fourier_series`
    """
    fs = make_fourier_series(df.shape[0], period, order=order, shift=shift)
    fs_cols = []
    for i in range(1, order + 1):
        fs_cols.append("{}fs_cos{}{}".format(prefix, i, suffix))
        fs_cols.append("{}fs_sin{}{}".format(prefix, i, suffix))
    fs_df = pd.DataFrame(fs, columns=fs_cols)
    out = pd.concat([df.reset_index(drop=True), fs_df], axis=1)
    return out, fs_cols


def make_seasonal_dummies(df, date_col, freq, sparse=True, drop_first=True):
    """Based on the frequency input (in pandas.DataFrame style), provide dummies indicator for regression type of
    purpose.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to supply datetime array for generating series of indicators
    date_col : str
        Label of the date column supply for generating series
    freq : str ['weekday', 'month', 'week']
        Options to pick the right frequency for generating dummies
    sparse : bool
        argument passed into `pd.get_dummies`
    drop_first : bool
        argument passed into `pd.get_dummies`
    Returns
    -------
    out : pd.DataFrame
        data with computed fourier series attached
    fs_cols : list
        list of labels derived from fourier series
    Notes
    -----
        This is calling :func:`pd.get_dummies`
    """
    if freq == "weekday":
        dummies = pd.get_dummies(
            df[date_col].dt.weekday, prefix="wd", sparse=sparse, drop_first=drop_first
        )
    elif freq == "month":
        dummies = pd.get_dummies(
            df[date_col].dt.month, prefix="m", sparse=sparse, drop_first=drop_first
        )
    elif freq == "week":
        dummies = pd.get_dummies(
            df[date_col].dt.week, prefix="w", sparse=sparse, drop_first=drop_first
        )
    else:
        raise IllegalArgument("Invalid argument of freq.")

    cols = dummies.columns.tolist()
    out = pd.concat([df, dummies], axis=1)
    return out, cols


def make_seasonal_regressors(n, periods, orders, labels, shift=0):
    """

    Parameters
    ----------
    n : int
        total length of time steps to generate seasonality; e.g. can simply be the length of your date array or
        dataframe
    periods : list
        list of period (a.k.a seasonality)
    orders : list
        list of fourier series order; needs to be the same length of s
    labels : list
        list of string to label each component
    shift : int
        shift of time step/index to generate the series

    Returns
    -------
    dict :
        a dictionary contains sine-cosine like regressors where keys are mapped by label provided
    """
    out = dict()
    for idx, period in enumerate(periods):
        order = orders[idx]
        label = labels[idx]
        fs = make_fourier_series(n=n, period=period, order=order, shift=shift)
        out[label] = fs
    return out


def moving_average(x, window=1, mode="same"):
    """Compute moving average of a 1-D numpy array

    Parameters
    ----------
    x : array_like
        1D-like array required to compute moving average
    window : int
        window size to compute moving average of the array
    mode : str
        one of the ['full', 'same', valid']. See `numpy.convolve` for details.
    Returns
    -------
        moving average of a 1-D array
    """
    return np.convolve(x, np.ones(window), mode=mode) / window
