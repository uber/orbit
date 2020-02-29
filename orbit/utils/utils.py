import os
# the following lines are added to fix unit test error
# or else the following line will give the following error
# TclError: no display name and no $DISPLAY environment variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import OrderedDict


if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


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


def plot_predicted_data(training_actual_df, predicted_df, date_col, actual_col, pred_col,
                        title="", test_actual_df=None, pred_quantiles_col=[],
                        is_visible=True):
    """
    plot training actual response together with predicted data; if actual response of predicted
    data is there, plot it too.
    Parameters
    ----------
    training_actual_df: pd.DataFrame
        training actual response data frame. two columns required: actual_col and date_col
    predicted_df: pd.DataFrame
        predicted data response data frame. two columns required: actual_col and pred_col. If
        user provide pred_quantiles_col, it needs to include them as well.
    date_col: str
        the date column name
    actual_col: str
    pred_col: str
    title: str
        title of the plot
    test_actual_df: pd.DataFrame
       test actual response dataframe. two columns required: actual_col and date_co
    pred_quantiles_col: list
        a list of two strings for prediction inference where first one for lower quantile and
        the second one for upper quantile
    is_visible: boolean
        whether we want to show the plot. If called from unittest, is_visible might = False.
    Returns
    -------
        None.

    """

    if is_empty_dataframe(training_actual_df) or is_empty_dataframe(predicted_df):
        raise ValueError("No prediction data or training response to plot.")
    if len(pred_quantiles_col) != 2 and len(pred_quantiles_col) != 0:
        raise ValueError("pred_quantiles_col must be either empty or length of 2.")
    if not set([pred_col] + pred_quantiles_col).issubset(predicted_df.columns):
        raise ValueError("Prediction column(s) not found in predicted df.")
    _training_actual_df=training_actual_df.copy()
    _predicted_df=predicted_df.copy()
    _training_actual_df[date_col] = pd.to_datetime(_training_actual_df[date_col])
    _predicted_df[date_col] = pd.to_datetime(_predicted_df[date_col])

    plt.figure(figsize=(16, 8))

    plt.scatter(_training_actual_df[date_col].values,
                _training_actual_df[actual_col].values,
                marker='.', color='black', alpha=0.5, s=70.0,
                label=actual_col)
    plt.plot(_predicted_df[date_col].values,
             _predicted_df[pred_col].values,
             marker=None, color='#12939A', label='prediction')

    if test_actual_df is not None:
        test_actual_df=test_actual_df.copy()
        test_actual_df[date_col] = pd.to_datetime(test_actual_df[date_col])
        plt.scatter(test_actual_df[date_col].values,
                    test_actual_df[actual_col].values,
                    marker='.', color='#FF8C00', alpha=0.5, s=70.0,
                    label=actual_col)

    # prediction intervals
    if pred_quantiles_col:
        plt.fill_between(_predicted_df[date_col].values,
                         _predicted_df[pred_quantiles_col[1]].values,
                         _predicted_df[pred_quantiles_col[0]].values,
                         facecolor='#42999E', alpha=0.5)

    plt.suptitle(title, fontsize=16)
    plt.legend()
    if is_visible:
        plt.show()


def get_parent_path(current_file_path):
    """

    Parameters
    ----------
    current_file_path: str
        The given file path, should be an absolute path

    Returns: str
        The parent path of give file path
    -------

    """

    return os.path.abspath(os.path.join(current_file_path, os.pardir))


def vb_extract(results):
    """Re-arrange and extract posteriors from variational inference fit from stan

    Due to different structure of the output from fit from vb, we need this additional logic to
    extract posteriors.  The logic is based on
    https://gist.github.com/lwiklendt/9c7099288f85b59edc903a5aed2d2d64

    Args
    ----
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


