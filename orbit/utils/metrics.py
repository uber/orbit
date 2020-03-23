import numpy as np
EPS = 1e-5

# utilities/loss to evaluate model performance

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


def mape(actual, predicted):
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
    return np.mean(np.abs((actual - predicted) / actual))


def wmape(actual, predicted):
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
    return np.sum(weights * np.abs((actual - predicted) / actual))


def mse(actual, predicted):
    ''' calculate mse metric

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        mse value
    '''

    return np.mean(np.square(actual - predicted))


def mse_naive(observed, exclude_leading_zeros=True):
    ''' calculate mse using last observed value as predictor
    Parameters
    ----------
    actual: array-like
        observed values
    Returns
    -------
    float
        mse value
    '''
    actual = observed[1:]
    predicted = observed[:-1]
    if exclude_leading_zeros:
        first_non_zero = np.min(np.nonzero(predicted))
        actual = actual[first_non_zero:]
        predicted = predicted[first_non_zero:]

    return mse(actual, predicted)
