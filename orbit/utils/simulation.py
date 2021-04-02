import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import math
from orbit.exceptions import IllegalArgument


def make_trend(series_len,  method='rw', arma=[.25, .6], rw_loc=0.0, rw_scale=0.1, seed=1):
    """ Module to generate time-series trend with different methods
    Parameters
    ----------
    series_len: int
        Total length of series
    method: str ['arma', 'rw']
        In case of `'rw'`, a simple random walk process will be used. For `'arma'`, we will use `statsmodels.api` to
        simulate a simple ARMA(1, 1) process
    arma: list
        List [arparams, maparams] of size 2 where used for arma(1) generating process
    rw_loc: float
        Location parameter of random walk generated by `np.random.normal()`
    rw_scale: float
        Scale parameter of random walk generated by `np.random.normal()`
    seed: int
        Seed passed into `np.random.default_rng()`
    Returns
    -------
    np.array-llike
        Simulated trend with length equals `series_len`

    Notes
    -----
        1. ARMA process: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.html
    """
    # make trend
    if method == "rw":
        rw = np.random.default_rng(seed).normal(rw_loc, rw_scale, series_len)
        trend = np.cumsum(rw)
    elif method == "arma":
        arparams = np.array([arma[0]])
        maparams = np.array([arma[1]])
        # add zero-lag and negate
        ar = np.r_[1, -arparams]
        # add zero-lag
        ma = np.r_[1, maparams]
        arma_process = ArmaProcess(ar, ma)
        trend = arma_process.generate_sample(series_len)
    else:
        raise IllegalArgument("Invalid trend method.")

    return trend


def make_seasonality(series_len, seasonality, method='discrete', order=3, duration=1, scale=.05, seed=1):
    """ Module to generate time-series seasonality with different methods
    series_len: int
        Total length of series
    seasonality: int
        For example, seasonality=52 would be a weekly series
    method: str ['discrete', 'fourier']
    order: int
        Used when method = 'fourier' ONLY. Fourier series order to generate seasonality.
    duration: int
        Used when method = 'discrete'. For example, seasonality=52 and duration=7 would be a daily
        series with a constant level on each week.
    scale: float
        Scale parameter of seasonality generation by Normal(0, scale). When method = 'discrete', it is directly used.
        when method = 'fourier', it is used to multiply with the sin-cosine wave generated series.
    seed: int
        Seed passed into `np.random.default_rng()`
    Returns
    -------
    np.array-llike
        Simulated seasonliaty with length equals `series_len`

    Notes
    -----
      1. In case of method = 'discrete', only `seasonality - 1` variables will be generated and the rest is
      directly derived to preserve the property of the sum of seasonality equals 1.
      2.  In case of method = 'fourier', see https://otexts.com/fpp2/complexseasonality.html
    """
    if seasonality > 1:
        if method == 'fourier':
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


def make_regression(series_len, coefs, loc=0.0, scale=0.5, cov=None, noise_scale=1.0,
                    relevance=1.0, sparsity=0.2, seed=1):
    """ Module to generate multiplicative time-series with trend, seasonality and regression components
    Parameters
    ----------
    series_len: int
        Total length of series
    coefs: 1-D array_like
        Values used as regression coefficients
    loc: float
        Location parameter to generate regressor values
    scale: float
        Scale parameter to generate regressor values
    cov: 2-D array_like, of shape (num_of_regressors, num_of_regressors)
        Covariance of regressors; need to be positive definite
    noise_scale:
        Scale parameter in the white noise generation process
    relevance: float
        0 to 1; smaller value indicates less number of total useful regressors
    sparsity: float
        0 to 1 to control probability (= 1 - sparsity) at time t of a regressor value > 0
    seed: int
        Seed passed into `np.random.default_rng()`

    Returns
    -------
    x: 2-D array like
        Regressors simulated array. Should be with shape (series_len, num_of_regressors);
    y: 1-D array like
        Regression derived by product of X and coefficients plus noise
    coefs: float
        Coefficients modified in the process due to sparsity; if sparsity=0, it should be
        identical to coefs inputted by user
    """

    num_of_regressors = len(coefs)
    if (relevance > 0.0) and (relevance < 1.0):
        num_of_irr_coefs = int(num_of_regressors * (1 - relevance))
        coefs = np.copy(coefs)
        irr_idx = np.random.choice(num_of_regressors, num_of_irr_coefs, replace=False)
        coefs[irr_idx] = 0.0

    if cov is None:
        x = np.random.default_rng(seed).normal(
            loc,
            scale,
            series_len * num_of_regressors
        ).reshape(series_len, -1)
    else:
        x = np.random.default_rng(seed).multivariate_normal(
            np.array([loc] * num_of_regressors, dtype=np.float64),
            cov,
            series_len
        )
    # control probability of regression kick-in
    if (sparsity > 0.0) and (sparsity < 1.0):
        z = np.random.default_rng(seed).binomial(
            1,
            1 - sparsity,
            series_len * num_of_regressors
        ).reshape(series_len, -1)
        x = x * z

    noise = np.random.default_rng(seed).normal(0, noise_scale, series_len)
    # make observed response
    y = np.matmul(x, coefs) + noise
    return x, y, coefs


# def make_ts_multiplicative(series_len=200, seasonality=-1, coefs=None, regressor_relevance=0.0,
#                            regressor_log_loc=0.0, regressor_log_scale=0.2,
#                            regressor_log_cov=None,
#                            noise_to_signal_ratio=1.0, regression_sparsity=0.5,
#                            obs_val_base=1000, regresspr_val_base=1000,
#                            trend_type='rw', rw_loc=0.001, rw_scale=0.1, seas_scale=.05,
#                            response_col='y', seed=0):
#     """ [Deprecated] Module to generate multiplicative time-series with trend, seasonality and regression components
#     Parameters
#     ----------
#     series_len: int
#     seasonality: int
#     coefs: 1-D array_like for regression coefs
#     regressor_relevance: float
#         0 to 1; higher value indicates less number of useful regressors
#     regressor_log_loc: float
#     regressor_log_scale: float
#     regressor_log_cov: 2-D array_like, of shape (num_of_regressors, num_of_regressors)
#         covariance of regressors in log unit scale
#     noise_to_signal_ratio: float
#     regression_sparsity: float
#         0 to 1 to control probability of value > 0 at time t of a regressor
#     obs_val_base: float
#         positive values
#     regresspr_val_base: float
#         positive values
#     trend_type: str
#         ['arma', 'rw']
#     seas_scale: float
#     response_col: str
#     seed: int
#     Notes
#     ------
#         Some ideas are from https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html
#         and https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
#     """
#     with_regression = False
#     if coefs is not None:
#         with_regression = True
#     # make regression
#     if with_regression:
#         num_of_regressors = len(coefs)
#         num_irrelevant_coefs = int(num_of_regressors * regressor_relevance)
#         if num_irrelevant_coefs >= 1:
#             irrelevant_coef_idx = np.random.choice(num_of_regressors, num_irrelevant_coefs, replace=False)
#             coefs[irrelevant_coef_idx] = 0.0
#         if regressor_log_cov is None:
#             x_log1p = np.random.default_rng(seed).normal(
#                 regressor_log_loc, regressor_log_scale, series_len * num_of_regressors).reshape(series_len, -1) + 1
#         else:
#             x_log1p = np.random.default_rng(seed).multivariate_normal(
#                 np.array([regressor_log_loc] * num_of_regressors, dtype=np.float64),
#                 regressor_log_cov, series_len)
#         # control probability of regression kick-in
#         z = np.random.default_rng(seed).binomial(
#             1, regression_sparsity, series_len * num_of_regressors).reshape(series_len, -1)
#         x_obs = x_log1p * z
#
#     trend = make_trend(series_len=series_len, rw_loc=rw_loc, rw_scale=rw_scale, method=trend_type, seed=seed)
#     seas = make_seasonality(series_len=series_len, seasonality=seasonality, scale=seas_scale, seed=seed)
#
#     # make noise
#     obs_log_scale = noise_to_signal_ratio * regressor_log_scale
#     noise = np.random.default_rng(seed).normal(0, obs_log_scale, series_len)
#
#     # make observed data
#     if with_regression:
#         y = np.round(obs_val_base * np.exp(trend + seas + np.matmul(x_obs, coefs) + noise)).reshape(-1, 1)
#         X = np.round(np.expm1(x_obs) * regresspr_val_base)
#         observed_matrix = np.concatenate([y, X], axis=1)
#         regressor_cols = [f"regressor_{x}" for x in range(1, num_of_regressors + 1)]
#         df_cols = [response_col] + regressor_cols
#     else:
#         y = np.round(obs_val_base * np.exp(trend + seas + noise)).reshape(-1, 1)
#         observed_matrix = y
#         df_cols = [response_col]
#
#     # TODO: right now we hard-coded the frequency; it is not impactful since in orbit we are only using date_col
#     # TODO: as index
#     # datetime index
#     if seasonality == 52:
#         dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1W")
#     elif seasonality == 12:
#         dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1M")
#     else:
#         dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1D")
#
#     df = pd.DataFrame(observed_matrix, columns=df_cols)
#     df['date'] = dt
#
#     return df, trend, seas, coefs
