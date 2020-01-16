import numpy as np
EPS = 1e-5


# a few utility functions to calculate mape/wmape/smape metrics
def smape(actual, predicted):
    ''' calculate symmetric mape

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        symmetric mape value
    '''
    actual = actual[np.abs(actual) > EPS]
    predicted = predicted[np.abs(predicted) > EPS]
    return 2 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


def mape(actual, pred):
    ''' calculate mape metric

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        mape value
    '''
    actual = actual[np.abs(actual) > EPS]
    return np.mean(np.abs((actual - pred) / actual))


def wmape(actual, pred):
    ''' calculate weighted mape metric

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        wmape value
    '''
    actual = actual[np.abs(actual) > EPS]
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return np.sum(weights * np.abs((actual - pred) / actual))