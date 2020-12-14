import numpy as np
import pandas as pd
from .features import make_fourier_series_df


def reduce_by_max(x, n=2):
    out = x.copy()
    out[np.argsort(x)[:-n]] = 0
    return out


# Gaussian-Kernel
# https://en.wikipedia.org/wiki/Kernel_smoother
def gauss_kernel(x, x_i, rho=1.0, alpha=1.0, n_reduce=-1):
    """
    x: points required to compute kernel weight
    x_i: reference points location used to compute correspondent distance of each entry points
    rho: smoothing parameter known as "length-scale" in gaussian process
    alpha: marginal standard deviation parameter in gaussian process; one should ignore in kernel regression (keep it = 1.0)
    b[deprecated]: radius or sometime named as (2*rho) that controls strength of covariance; the smaller the shorter raidus (dist. to negihbour)
    will take into effect
    Returns
    -------
        2D array with N x M such that
        N as the number of entry points
        M as the number of reference points
        matrix entries hold the value of weight of each element
    see also:
      1. https://mc-stan.org/docs/2_24/stan-users-guide/gaussian-process-regression.html
      2. https://en.wikipedia.org/wiki/Local_regression
    """
    N = len(x)
    M = len(x_i)
    k = np.zeros((N, M), np.double)
    alpha_sq = alpha ** 2
    rho_sq_t2 = 2 * rho ** 2
    for n in range(N):
        k[n, :] = alpha_sq * np.exp(-1 * (x[n] - x_i) ** 2 / rho_sq_t2)

    if n_reduce > 0:
       k = np.apply_along_axis(reduce_by_max, axis=1, arr=k, n=n_reduce)

    k = k / np.sum(k, axis=1, keepdims=True)

    return k


def sandwich_kernel(x, x_i):
    """
    x: points required to compute kernel weight
    x_i: reference points location used to compute correspondent distance of each entry points
    rho: smoothing parameter known as "length-scale" in gaussian process
    alpha: marginal standard deviation parameter in gaussian process; one should ignore in kernel regression (keep it = 1.0)
    b[deprecated]: radius or sometime named as (2*rho) that controls strength of covariance; the smaller the shorter raidus (dist. to negihbour)
    will take into effect
    return:
        a matrix with N x M such that
        N as the number of entry points
        M as the number of reference points
        matrix entries hold the value of weight of each element
    see also:
      1. https://mc-stan.org/docs/2_24/stan-users-guide/gaussian-process-regression.html
      2. https://en.wikipedia.org/wiki/Local_regression
    """
    N = len(x)
    M = len(x_i)
    k = np.zeros((N, M), dtype=np.double)

    np_idx = np.where(x < x_i[0])
    k[np_idx, 0] = 1

    for m in range(M - 1):
        np_idx = np.where(np.logical_and(x >= x_i[m], x < x_i[m + 1]))
        total_dist = x_i[m + 1] - x_i[m]
        backward_dist = x[np_idx] - x_i[m]
        forward_dist = x_i[m + 1] - x[np_idx]
        k[np_idx, m] = forward_dist / total_dist
        k[np_idx, m + 1] = backward_dist / total_dist

    np_idx = np.where(x >= x_i[M - 1])
    k[np_idx, M - 1] = 1

    # TODO: it is probably not needed
    k = k / np.sum(k, axis=1, keepdims=True)

    return k

def generate_tp(prediction_date_array, training_df_meta):
    # should be you prediction date array
    prediction_start = prediction_date_array[0]
    trained_len = training_df_meta['df_length']
    output_len = len(prediction_date_array)
    if prediction_start > training_df_meta['training_end']:
        start = trained_len
    else:
        start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

    new_tp = np.arange(start + 1, start + output_len + 1) / trained_len
    return new_tp


def generate_insample_tp(date_array, training_df_meta):
    idx = np.nonzero(np.in1d(training_df_meta['date_array'], date_array))[0]
    tp = (idx + 1) / training_df_meta['df_length']
    return tp


def generate_levs(prediction_date_array, training_df_meta, level_knot_dates, lev_knot):
    new_tp = generate_tp(prediction_date_array, training_df_meta)
    knots_tp_level = generate_insample_tp(level_knot_dates, training_df_meta)
    kernel_level = sandwich_kernel(new_tp, knots_tp_level)
    levs = np.matmul(lev_knot, kernel_level.transpose(1, 0))
    return levs


def generate_coefs(prediction_date_array, training_df_meta, coef_knot_dates, coef_knot, rho):
    new_tp = generate_tp(prediction_date_array, training_df_meta)
    knots_tp_coef = generate_insample_tp(coef_knot_dates, training_df_meta)
    kernel_coef = gauss_kernel(new_tp, knots_tp_coef, rho)
    kernel_coef = kernel_coef / np.sum(kernel_coef, axis=1, keepdims=True)
    coefs = np.squeeze(np.matmul(coef_knot, kernel_coef.transpose(1, 0)), axis=0).transpose(1, 0)
    return coefs


def generate_seas(df, date_col, training_df_meta, coef_knot_dates, coef_knot,
                  rho, seasonality, seasonality_fs_order):
    prediction_date_array = df[date_col].values
    prediction_start = prediction_date_array[0]
    trained_len = training_df_meta['df_length']
    df = df.copy()
    if prediction_start > training_df_meta['training_end']:
        forecast_dates = set(prediction_date_array)
        n_forecast_steps = len(forecast_dates)
        # time index for prediction start
        start = trained_len
    else:
        # compute how many steps to forecast
        forecast_dates = set(prediction_date_array) - set(training_df_meta['date_array'])
        # check if prediction df is a subset of training df
        # e.g. "negative" forecast steps
        n_forecast_steps = len(forecast_dates) or \
                           - (len(set(training_df_meta['date_array']) - set(prediction_date_array)))
        # time index for prediction start
        start = pd.Index(training_df_meta['date_array']).get_loc(prediction_start)

    fs_cols = []
    for idx, s in enumerate(seasonality):
        order = seasonality_fs_order[idx]
        df, fs_cols_temp = make_fourier_series_df(df, s, order=order, prefix='seas{}_'.format(s), shift=start)
        fs_cols += fs_cols_temp

    sea_regressor_matrix = df.filter(items=fs_cols).values
    sea_coefs = generate_coefs(prediction_date_array, training_df_meta, coef_knot_dates, coef_knot, rho)
    seas = np.sum(sea_coefs * sea_regressor_matrix, axis=-1)

    return seas