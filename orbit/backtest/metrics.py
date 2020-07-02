import numpy as np
EPS = 1e-5

# utilities/loss to evaluate model performance


def smape(outcome):
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

    actual = outcome['test_actual']
    predicted = outcome['test_pred']

    filtered = (np.abs(actual) > EPS) & (np.abs(predicted) > EPS)
    actual = actual[filtered]
    predicted = predicted[filtered]
    return 2 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


def mape(outcome):
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

    actual = outcome['test_actual']
    predicted = outcome['test_pred']

    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    return np.mean(np.abs((actual - predicted) / actual))


def wmape(outcome):
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

    actual = outcome['test_actual']
    predicted = outcome['test_pred']

    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return np.sum(weights * np.abs((actual - predicted) / actual))


def _mae(actual, predicted, exclude_leading_zeros=False):
    ''' calculate mean absolute error

    Parameters
    ----------
    actual: array-like
        true values
    predicted: array-like
        prediction values

    Returns
    -------
    float
        mae value
    '''

    if exclude_leading_zeros:
        first_nz = np.min(np.flatnonzero(actual))
        actual = actual[first_nz:]
        predicted = predicted[first_nz:]

    return np.mean(np.abs(actual - predicted))


def _mse(actual, predicted, exclude_leading_zeros=False):
    ''' calculate mean squared error

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
        first_nz = np.min(np.flatnonzero(actual))
        # print(first_nz)
        actual = actual[first_nz:]
        predicted = predicted[first_nz:]

    return np.mean(np.square(actual - predicted))


def rmsse(outcome):
    ''' Root Mean Squared Scaled Error (RMSSE), a variant of the well-known Mean Absolute Scaled Error (MASE)
    proposed by Hyndman and Koehler (2006)

    .. math::
        \sqrt{\frac{1}{h}\frac{\sum^{n+h}_{t=n+1}(Y_t-\hat{Y}_t)^2}{\frac{1}{n-1}\sum^{n}_{t=2}{(Y_t-Y_{t-1})^2}

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

    test_actual = outcome['test_actual']
    test_pred = outcome['test_pred']
    train_actual = outcome['train_actual']

    lag1_mse = _mse(train_actual[1:], train_actual[:-1], exclude_leading_zeros=True)
    # print(lag1_mse)
    # print(train_actual)
    forecast_mse = _mse(test_actual, test_pred, exclude_leading_zeros=False)
    return np.sqrt(forecast_mse / lag1_mse)
