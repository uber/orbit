import numpy as np
EPS = 1e-5


def smape(actual, predicted):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    return 2 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


def mape(actual, predicted):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    return np.mean(np.abs((actual - predicted) / actual))


def wmape(actual, predicted):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    predicted = predicted[filtered]
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return np.sum(weights * np.abs((actual - predicted) / actual))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def mse(actual, predicted):
    return np.mean(np.square(actual - predicted))


def rmsse(test_actual, test_predicted, train_actual):
    """Computes Root Mean Squared Scaled Error (RMSSE)

    A variant of the well-known Mean Absolute Scaled Error (MASE) proposed by Hyndman and Koehler (2006)

    Intuitively, the general idea is to measure additional value added by the model
    compared to a naive lag-1 predictor.

    .. math::
        \sqrt{\frac{1}{h}\frac{\sum^{n+h}_{t=n+1}(Y_t-\hat{Y}_t)^2}{\frac{1}{n-1}\sum^{n}_{t=2}{(Y_t-Y_{t-1})^2}

    Notes
    -----
    Reference from https://mofc.unic.ac.cy/m5-guidelines/

    """

    # exclude leading zeros for training
    first_nz = np.min(np.flatnonzero(train_actual))
    train_actual = test_actual[first_nz:]

    lag1_mse = mse(train_actual[1:], train_actual[:-1])

    forecast_mse = mse(test_actual, test_predicted)

    return np.sqrt(forecast_mse / lag1_mse)
