import os

# the following lines are added to fix unit test error
# or else the following line will give the following error
# TclError: no display name and no $DISPLAY environment variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uTS.utils.utils_base import (
    is_empty_dataframe)
from uTS.utils.constants import (
    PlotLabels,
    PredictionColumnNames,
    RegressionStanSamplingParameters)
from uTS.utils.measure import mean_absolute_percentage_error

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


# def quantile(q):
#     quantile_funcs=[]
#     for qq in q:
#         n=int(qq*100)
#         def _quantile(x):
#             return np.percentile(x, n)
#         _quantile.__name__ = 'quantile{0:02d}'.format(n)
#         quantile_funcs.append(_quantile)
#     return quantile_funcs


# def get_model_regressor_beta(model_parameters):
#     """
#     Concatenate `positive_regressor_beta`s and `regular_regressor_beta`s from model object
#
#     :param model_parameters: dict or list of dict
#         model_parameters object
#     :return: If mcmc return a 2d array of length (num_samples, pr_betas + rr_betas)
#     """
#
#     if isinstance(model_parameters, dict):
#
#         positive_regressor_beta = model_parameters \
#             .get(RegressionStanSamplingParameters.POSITIVE_REGRESSOR_BETA.value, np.array([]))
#
#         regular_regressor_beta = model_parameters \
#             .get(RegressionStanSamplingParameters.REGULAR_REGRESSOR_BETA.value, np.array([]))
#
#         regressor_beta = np.concatenate((positive_regressor_beta, regular_regressor_beta))
#
#         return regressor_beta
#
#     elif isinstance(model_parameters, list):
#
#         regressor_betas = []
#         for params in model_parameters:
#
#             regressor_beta = get_model_regressor_beta(params)
#             regressor_betas.append(regressor_beta)
#
#         return regressor_betas
#
#     else:
#         raise TypeError("Error: Model object must be a dict or list of dicts")


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
        # plt.plot(
        #     _predicted_df[date_col].values,
        #     _predicted_df[5].values,
        #     linestyle=':', color='blue', label='Lower Prediction Interval'
        # )
        # plt.plot(
        #     _predicted_df[date_col].values,
        #     _predicted_df[95].values,
        #     linestyle=':', color='blue', label='Upper Prediction Interval'
        # )
        # plt.plot(
        #     predicted_df_quantiles[date_col].values.ravel(),
        #     predicted_df_quantiles[
        #         'pred_05'
        #     ].iloc[:, 0].values,
        #     linestyle=':', color='blue', label='Lower Prediction Interval'
        # )
        # plt.plot(
        #     predicted_df_quantiles[date_col].values.ravel(),
        #     predicted_df_quantiles[
        #         'pred_95'
        #     ].iloc[:, 1].values,
        #     linestyle=':', color='blue', label='Upper Prediction Interval'
        # )

    # if pred_col in predicted_df.columns:
    #     plt.scatter(predicted_df[date_col].values,
    #                 predicted_df[pred_col].values,
    #                 marker='+', color='red', alpha=0.5, label=PlotLabels.ACTUAL_RESPONSE.value)
    #     mape = mean_absolute_percentage_error(
    #         predicted_df[pred_col],
    #         predicted_df[pred_col])
    #     title = "{}, MAPE = {}".format(title, '%.3f' % mape)

    plt.suptitle(title, fontsize=16)
    plt.legend()
    if is_visible:
        plt.show()


def is_ordered_datetime(array):
    """Returns True if array is ordered and non-repetitive"""
    return np.all(np.diff(array).astype(float) > 0)
