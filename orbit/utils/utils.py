import os
# the following lines are added to fix unit test error
# or else the following line will give the following error
# TclError: no display name and no $DISPLAY environment variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from copy import deepcopy

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


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