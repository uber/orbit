import numpy as np
import pandas as pd
from .features import make_fourier_series_df


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