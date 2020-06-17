import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from orbit.exceptions import IllegalArgument


def make_trend(series_len, rw_loc=0.001, rw_scale=0.1, type='rw', seed=1):
    """ Module to generate time-series trend with different methods

    Parameters
    ----------
    series_len: int
        total length of series
    type: str ['arma', 'rw']
    rw_loc: float
        mean of random walk
    rw_scale: float
        scale of random walk
    seed: int
    Returns
    -------
      np.array-llike with length equals `series_len`
    """
    # make trend
    if type == "rw":
        rw = np.random.default_rng(seed).normal(rw_loc, rw_scale, series_len)
        trend = np.cumsum(rw)
    elif type == "arma":
        # TODO: consider parameterize this
        arparams = np.array([.25])
        maparams = np.array([.6])
        ar = np.r_[1, -arparams]
        ma = np.r_[1, maparams]
        arma_process = sm.tsa.ArmaProcess(ar, ma)
        trend = arma_process.generate_sample(series_len)
    else:
        raise IllegalArgument("Invalid trend_type.")

    return trend


def make_seasonality(series_len, seasonality, order=3, duration=1, scale=.05, type='discrete', seed=1):
    """ Module to generate time-series seasonality with different methods
    series_len: int
        total length of series
    seasonality: int
        for example, seasonality=52 would be a weekly series
    order: int
        fourier series order to generate seasonality.  Used when type = 'fourier' ONLY.
    duration: int
        for example, seasonality=52 and duration=7 would be a daily series with annual seasonality on weeks. Used
        in non-fourier type of seasonality ONLY.
    scale: float
        scale parameter of seasonality generation
    type: str ['discrete', 'fourier']
    seed: int
    Returns
    -------
      np.array-llike with length equals `series_len`
    """
    if seasonality > 1:
        if type == 'fourier':
            t = np.arange(0, series_len)
            out = []
            for i in range(1, order + 1):
                x = 2.0 * i * np.pi * t / seasonality
                out.append(np.sin(x))
                out.append(np.cos(x))
            out = np.column_stack(out)
            b = np.random.default_rng(seed).normal(0, scale, order * 2)
            seas = np.matmul(out, b)
        else:
            # initialization
            seas = []
            iterations = math.ceil(series_len / duration)
            # initialize vector to be repeated
            init_seas = np.zeros(seasonality)
            init_seas[:-1] = np.random.default_rng(seed).normal(0, scale, seasonality - 1)
            init_seas[seasonality - 1] = -1 * np.sum(init_seas)
            for idx in range(iterations):
                seas += [init_seas[idx % seasonality]] * duration
            seas = np.array(seas[:series_len])
    else:
        seas = np.zeros(series_len)
    return seas


def make_ts_multiplicative(series_len=200, seasonality=-1, coefs=None, regressor_relevance=0.0,
                           regressor_log_loc=0.0, regressor_log_scale=0.2,
                           regressor_log_cov=None,
                           noise_to_signal_ratio=1.0, regression_sparsity=0.5,
                           obs_val_base=1000, regresspr_val_base=1000,
                           trend_type='rw', rw_loc=0.001, rw_scale=0.1, seas_scale=.05,
                           response_col='y', seed=0):
    """
    Parameters
    ----------
    series_len: int
    seasonality: int
    coefs: 1-D array_like for regression coefs
    regressor_relevance: float
        0 to 1; higher value indicates less number of useful regressors
    regressor_log_loc: float
    regressor_log_scale: float
    regressor_log_cov: 2-D array_like, of shape (num_of_regressors, num_of_regressors)
        covariance of regressors in log unit scale
    noise_to_signal_ratio: float
    regression_sparsity: float
        0 to 1 to control probability of value > 0 at time t of a regressor
    obs_val_base: float
        positive values
    regresspr_val_base: float
        positive values
    trend_type: str
        ['arma', 'rw']
    seas_scale: float
    response_col: str
    seed: int

    Notes
    ------
        Some ideas are from https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html
        and https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
    """
    with_regression = False
    if coefs is not None:
        with_regression = True
    # make regression
    if with_regression:
        num_of_regressors = len(coefs)
        num_irrelevant_coefs = int(num_of_regressors * regressor_relevance)
        if num_irrelevant_coefs >= 1:
            irrelevant_coef_idx = np.random.choice(num_of_regressors, num_irrelevant_coefs, replace=False)
            coefs[irrelevant_coef_idx] = 0.0
        if regressor_log_cov is None:
            x_log1p = np.random.default_rng(seed).normal(
                regressor_log_loc, regressor_log_scale, series_len * num_of_regressors).reshape(series_len, -1) + 1
        else:
            x_log1p = np.random.default_rng(seed).multivariate_normal(
                np.array([regressor_log_loc] * num_of_regressors, dtype=np.float64),
                regressor_log_cov, series_len)
        # control probability of regression kick-in
        z = np.random.default_rng(seed).binomial(
            1, regression_sparsity, series_len * num_of_regressors).reshape(series_len, -1)
        x_obs = x_log1p * z

    # make trend
    # if trend_type == "rw":
    #     rw = np.random.default_rng(seed).normal(rw_loc, rw_scale, series_len)
    #     trend = np.cumsum(rw)
    # elif trend_type == "arma":
    #     # TODO: consider parameterize this
    #     arparams = np.array([.25])
    #     maparams = np.array([.6])
    #     ar = np.r_[1, -arparams]
    #     ma = np.r_[1, maparams]
    #     arma_process = sm.tsa.ArmaProcess(ar, ma)
    #     trend = arma_process.generate_sample(series_len)
    # else:
    #     raise IllegalArgument("Invalid trend_type.")
    trend = make_trend(series_len=series_len, rw_loc=rw_loc, rw_scale=rw_scale, type=trend_type, seed=seed)

    # make seasonal component
    # if seasonality > 1:
    #     init_seas = np.zeros(seasonality)
    #     init_seas[:-1] = np.random.default_rng(seed).normal(0, seas_scale, seasonality - 1)
    #     init_seas[seasonality - 1] = -1 * np.sum(init_seas)
    #     seas = np.zeros(series_len)
    #     for idx in range(series_len):
    #         seas[idx] = init_seas[idx % seasonality]
    # else:
    #     seas = np.zeros(series_len)
    seas = make_seasonality(seasonality=seasonality, series_len=series_len, scale=seas_scale, seed=seed)

    # make noise
    obs_log_scale = noise_to_signal_ratio * regressor_log_scale
    noise = np.random.default_rng(seed).normal(0, obs_log_scale, series_len)

    # make observed data
    if with_regression:
        y = np.round(obs_val_base * np.exp(trend + seas + np.matmul(x_obs, coefs) + noise)).reshape(-1, 1)
        X = np.round(np.expm1(x_obs) * regresspr_val_base)
        observed_matrix = np.concatenate([y, X], axis=1)
        regressor_cols = [f"regressor_{x}" for x in range(1, num_of_regressors + 1)]
        df_cols = [response_col] + regressor_cols
    else:
        y = np.round(obs_val_base * np.exp(trend + seas + noise)).reshape(-1, 1)
        observed_matrix = y
        df_cols = [response_col]

    # TODO: right now we hard-coded the frequency; it is not impactful since in orbit we are only using date_col
    # TODO: as index
    # datetime index
    if seasonality == 52:
        dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1W")
    elif seasonality == 12:
        dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1M")
    else:
        dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1D")

    df = pd.DataFrame(observed_matrix, columns=df_cols)
    df['date'] = dt

    return df, trend, seas, coefs
