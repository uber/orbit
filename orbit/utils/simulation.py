import pandas as pd
import numpy as np
import statsmodels.api as sm

from orbit.exceptions import IllegalArgument


def make_ts_multiplicative_regression(series_len=200, seasonality=-1, num_of_regressors=10, regressor_sparsity=0.0,
                                      coef_mean=0.0, coef_sd=.1, regressor_log_loc=0.0, regressor_log_scale=0.2,
                                      noise_to_signal_ratio=1.0, regression_prob=0.5,
                                      obs_val_base=1000, regresspr_val_base=1000, trend_type='rw',
                                      rw_loc=0.001, rw_scale=0.1,
                                      seas_scale=.05, response_col='y', seed=0):
    """
    Parameters
    ----------
        series_len: int
        seasonality: int
        num_of_regressors: int
        regressor_sparsity: float
            0 to 1; higher value indicates less number of useful regressors
        coef_mean: float
        coef_sd: float
        regressor_log_loc: float
        regressor_log_scale: float
        noise_to_signal_ratio: float
        regressorion_prob: float
            0 to 1
        obs_val_base: float
            positive values
        regresspr_val_base: float
            positive values
        trend_type: str
            ['arma', 'rw']
        rw_loc: real
        rw_scale: real
        seas_scale: float
        response_col: str
        seed: int

    Notes
    ------
        Some ideas are from https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html
        and https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
    """
    coefs = np.random.default_rng(seed).normal(coef_mean, coef_sd, num_of_regressors)
    num_irrelevant_coefs = int(num_of_regressors * regressor_sparsity)
    if num_irrelevant_coefs >= 1:
        irrelevant_coef_idx = np.random.choice(num_of_regressors, num_irrelevant_coefs)
        coefs[irrelevant_coef_idx] = 0.0

    obs_log_scale = noise_to_signal_ratio * regressor_log_scale
    x_log1p = np.random.default_rng(seed).normal(
        regressor_log_loc, regressor_log_scale, series_len * num_of_regressors).reshape(series_len, -1) + 1
    # control probability of regression kick-in
    z = np.random.default_rng(seed).binomial(1, regression_prob, series_len * num_of_regressors).reshape(series_len, -1)
    x_obs = x_log1p * z
    noise = np.random.default_rng(seed).normal(0, obs_log_scale, series_len)

    if trend_type == "rw":
        rw = np.random.default_rng(seed).normal(rw_loc, rw_scale, series_len)
        trend = np.cumsum(rw)
    elif trend_type == "arma":
        arparams = np.array([.25])
        maparams = np.array([.6])
        ar = np.r_[1, -arparams]
        ma = np.r_[1, maparams]
        arma_process = sm.tsa.ArmaProcess(ar, ma)
        trend = arma_process.generate_sample(series_len)
    else:
        raise IllegalArgument("Invalid trend_type.")

    if seasonality > 1:
        init_seas = np.zeros(seasonality)
        init_seas[:-1] = np.random.default_rng(seed).normal(0, seas_scale, seasonality - 1)
        init_seas[seasonality - 1] = -1 * np.sum(init_seas)
        seas = np.zeros(series_len)
        for idx in range(series_len):
            seas[idx] = init_seas[idx % seasonality]
    else:
        seasonality = 1
        seas = np.zeros(series_len)

    y = np.round(obs_val_base * np.exp(trend + seas + np.matmul(x_obs, coefs) + noise))
    # unsqueeze to 2D
    y = y.reshape(-1, 1)
    X = np.round(np.expm1(x_obs) * regresspr_val_base)

    # TODO: right now we hard-coded the frequency; it is not impactful since in orbit we are only using date_col
    # TODO: as index
    # datetime index
    if seasonality == 52:
        dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1W")
    elif seasonality == 12:
        dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1M")
    else:
        dt = pd.date_range(start='2016-01-04', periods=series_len, freq="1D")

    regressor_cols = [f"regressor_{x}" for x in range(1, num_of_regressors + 1)]
    df = pd.DataFrame(np.concatenate([y, X], axis=1), columns=[response_col] + regressor_cols)
    df['date'] = dt

    return df, coefs, trend, seas
