import numpy as np
EPS = 1e-5

# utilities/loss to evaluate model performance


def smape(actual, predicted):
    ''' calculate symmetric mape

    Parameters
    ----------
    actual: array-like
        true values
    predicted: array-like
        prediction values

    Returns
    -------
    float
        symmetric mape value
    '''
    filtered = (np.abs(actual) > EPS) & (np.abs(predicted) > EPS)
    actual = actual[filtered]
    predicted = predicted[filtered]
    return 2 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


def mape(actual, predicted):
    ''' calculate mape metric

    Parameters
    ----------
    actual: array-like
        true values
    predicted: array-like
        prediction values

    Returns
    -------
    float
        mape value
    '''
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    return np.mean(np.abs((actual - predicted) / actual))


def wmape(actual, predicted):
    ''' calculate weighted mape metric

    Parameters
    ----------
    actual: array-like
        true values
    predicted: array-like
        prediction values

    Returns
    -------
    float
        wmape value
    '''
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return np.sum(weights * np.abs((actual - predicted) / actual))


def mse(actual, predicted, exclude_leading_zeros=True):
    ''' calculate mse metric

    Parameters
    ----------
    actual: array-like
        true values
    predicted: array-like
        prediction values

    Returns
    -------
    float
        mse value
    '''

    if exclude_leading_zeros:
        first_nz = np.min(np.nonzero(predicted))
        actual = actual[first_nz:]
        predicted = predicted[first_nz:]

    return np.mean(np.square(actual - predicted))


def rmsse(actual, predicted, n=None, exclude_leading_zeros=True):
    ''' Root Mean Squared Scaled Error (RMSSE), a variant of the well-known Mean Absolut Scaled Error (MASE)
    proposed by Hyndman and Koehler (2006)

    .. math::
        \sqrt{\frac{1}{h}\frac{\sum^{n+h}_{t=n+1}(Y_t-\hat{Y}_t)^2}{}

    Parameters
    ----------
    actual: array-like
        true values
    predicted: array-like
        prediction values
    n: int
        forecast periods

    Notes
    -----
    The general idea is that we want to measure additional value added by the model
    comparing to a naive lag-1 predictor.
    Reference from https://mofc.unic.ac.cy/m5-guidelines/
    '''
    if n is None:
        n = len(predicted)
    lag1_mse = mse(actual[1:-n], actual[:(-n-1)], exclude_leading_zeros)
    forecast_mse = mse(actual[-n:], predicted[-n:])
    return np.sqrt(forecast_mse / lag1_mse)
