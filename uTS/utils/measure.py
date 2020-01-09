'''
Implements various metrics to evaluate forecast accuracy, follows sklearn model API definition

'''
import numpy as np
import pandas as pd
from sklearn import metrics

__all__ = [
    "score",
    "performance"
]


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE residual error.
    :param y_true: np.array, the ground true values
    :param y_pred: np.array, the predicted values
    :return:
    -------
    float
        a positive value
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Symmetric MAPE residual error.
    :param y_true: np.array, the ground true values
    :param y_pred: np.array, the predicted values
    :return:
    -------
    float
        a positive value
    """
    y_true = y_true.astype(np.float)
    y_pred = y_pred.astype(np.float)
    return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def weighted_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate weighted MAPE.

    Parameters
    ----------
    y_true: 1d numpy array
    y_pred: 1d numpy array

    Returns
    -------
    score: float
    """
    y_true = y_true.astype(np.float)
    y_pred = y_pred.astype(np.float)
    score = np.sum(abs(y_true - y_pred)) / np.sum(abs(y_true))
    return score


Residual = dict(r2=metrics.r2_score,
                mae=metrics.mean_absolute_error,
                mse=metrics.mean_squared_error,
                mape=mean_absolute_percentage_error)


def score(y_true, y_pred, metric=None):
    """
    Calculate the prediction metrics.
    :param y_true: np.array, the ground true values
    :param y_pred: np.array, the predicted values
    :param metric: str, metric method
    :return:
    -------
    float
        a metric value.
    """
    if metric is None:
        return mean_absolute_percentage_error(y_true, y_pred)
    else:
        return Residual[metric](y_true, y_pred)


def performance(y_true, y_pred):
    """
    Calculate the performance of prediction result.
    :param y_true:
    :param y_pred: np.array, the predicted values
    :return:
    -------
    pd.DataFrame
        scores for different metrics
    """
    res = []
    for method in Residual.keys():
        res.append(score(y_true, y_pred, metric=method))

    return pd.DataFrame({"Error": res}, index=Residual.keys())
