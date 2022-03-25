import numpy as np

EPS = 1e-5


def smape(actual, prediction):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    prediction = prediction[filtered]
    return 2 * np.mean(
        np.abs(actual - prediction) / (np.abs(actual) + np.abs(prediction))
    )


def mape(actual, prediction):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    prediction = prediction[filtered]
    return np.mean(np.abs((actual - prediction) / actual))


def wmape(actual, prediction):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    prediction = prediction[filtered]
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return np.sum(weights * np.abs((actual - prediction) / actual))


def wsmape(actual, prediction):
    filtered = np.abs(actual) > EPS
    actual = actual[filtered]
    prediction = prediction[filtered]
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return 2 * np.sum(
        weights * np.abs(actual - prediction) / (np.abs(actual) + np.abs(prediction))
    )


def mae(actual, prediction):
    filtered = ~np.isnan(actual)
    actual = actual[filtered]
    prediction = prediction[filtered]
    return np.mean(np.abs(actual - prediction))


def mse(actual, prediction):
    filtered = ~np.isnan(actual)
    actual = actual[filtered]
    prediction = prediction[filtered]
    return np.mean(np.square(actual - prediction))


def rmsse(test_actual, test_prediction, train_actual):
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

    forecast_mse = mse(test_actual, test_prediction)

    return np.sqrt(forecast_mse / lag1_mse)


def wbic():
    pass
    # TODO:
    # 1. convert all likelihood to target += syntax by reading stan manual
    # https://mc-stan.org/docs/2_28/stan-users-guide/index.html
    # 2. implement the watanabe wbic temperature sampling formula
    # 3. create a toy model with 3 regressors and 2 useless regressor
    # 4. create DLTs one with 1-5 regressor
    # 5. create the WBIC function and compute the stats on those DLT models
    # 6.[bonus] after all, think of a way to automate / generalize the way to add watanabe wbic
    # temperature logic in a functional way
