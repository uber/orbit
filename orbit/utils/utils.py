import os
import numpy as np
import pandas as pd
import string
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression


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


def vb_extract(results):
    """Re-arrange and extract posteriors from variational inference fit from stan

    Due to different structure of the output from fit from vb, we need this additional logic to
    extract posteriors.  The logic is based on
    https://gist.github.com/lwiklendt/9c7099288f85b59edc903a5aed2d2d64

    Parameters
    ----------
    results: dict
        dict exported from pystan.StanModel object by `vb` method

    Returns
    -------
    params: OrderedDict
        dict of arrays where each element represent arrays of samples (Index of Sample, Sample
        dimension 1, Sample dimension 2, ...)
    """
    param_specs = results['sampler_param_names']
    samples = results['sampler_params']
    n = len(samples[0])

    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
            idxs = [int(i) for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # create arrays
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape))) for name, shape in param_shapes.items()])

    # second pass, set arrays
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # -1 because pystan returns 1-based indexes for vb!
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params


def simulate_seasonal_term(periodicity, total_cycles, noise_std=1.,
                           harmonics=None):
    """Generates a seasonality term"""
    # https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_seasonal.html
    duration = periodicity * total_cycles
    assert duration == int(duration)
    duration = int(duration)
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = noise_std * np.random.randn((harmonics))
    gamma_star_jt = noise_std * np.random.randn((harmonics))

    total_timesteps = 100 * duration # Pad for burn in
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (gamma_jt[j - 1] * cos_j
                                 + gamma_star_jt[j - 1] * sin_j
                                 + noise_std * np.random.randn())
            gamma_star_jtp1[j - 1] = (- gamma_jt[j - 1] * sin_j
                                      + gamma_star_jt[j - 1] * cos_j
                                      + noise_std * np.random.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1
    wanted_series = series[-duration:] # Discard burn in

    return wanted_series


def make_synthetic_series(seed=0):
    """Generate synthetic data with regressors"""
    np.random.seed(seed)

    # simulate seasonality
    seasonality_term = simulate_seasonal_term(periodicity=52, total_cycles=4, harmonics=3)

    # scale data
    scaler = MinMaxScaler(feature_range=(np.max(seasonality_term), np.max(seasonality_term) * 2))
    seasonality_term = scaler.fit_transform(seasonality_term[:, None])

    # datetime index
    dt = pd.date_range(start='2016-01-04', periods=len(seasonality_term), freq='7D')

    # create df
    df = pd.DataFrame(seasonality_term, columns=['response'], index=dt).reset_index()
    df = df.rename(columns={'index': 'week'})

    # make regression
    X, y, coef = make_regression(n_samples=df.shape[0], n_features=6, n_informative=3, random_state=seed, coef=True)

    scaler = MinMaxScaler(feature_range=(1, np.max(X) * 2))
    X = scaler.fit_transform(X)
    X = X / 2

    n_samples, n_features = X.shape
    feature_names = list(string.ascii_letters[:n_features])
    X_df = pd.DataFrame(X, columns=feature_names)

    # join regression
    df = pd.concat((df, X_df), axis=1)

    # sum response with regression response
    df['response'] = df['response'] + y

    # linear trend
    df['response'] = df['response'] * np.linspace(5, 1, df.shape[0])

    return df, coef


def fourier_series(dates, period, order=3):
    """ Given dates array, cyclical period and order.  Return a set of fourier series.

    Parameters
    ----------
    dates: array-like date-stamp
    period: int
    order: int

    Returns
    -------
        2D array where each column is a specific order fourier series with sin() or cos().
    """
    t = np.array(
        (dates - datetime(1970, 1, 1))
            .dt.total_seconds()
            .astype(np.float)
    ) / (3600 * 24.)
    out = list()
    for i in range(1, order + 1):
        x = 2.0 * i * np.pi * t / period
        out.append(np.sin(x))
        out.append(np.cos(x))
    out = np.column_stack(out)
    return out


def fourier_series_df(df, date_col, period, order=3):
    """ Given a data-frame, cyclical period and order.  Return a set of fourier series.
    Parameters
    ----------
    df: pd.DataFrame
    date_col: str
    period: int
    order: int

    Returns
    -------
    df: pd.DataFrame
        data with computed fourier series attached
    fs_cols: list
        list of labels derived from fourier series
    """
    fs = fourier_series(df[date_col], period, order=order)
    fs_cols = []
    for i in range(1, order + 1):
        fs_cols.append('fs_cos{}'.format(i))
        fs_cols.append('fs_sin{}'.format(i))
    fs_df = pd.DataFrame(fs, columns=fs_cols)
    df = pd.concat([df.reset_index(drop=True), fs_df], axis=1)
    return df, fs_cols
