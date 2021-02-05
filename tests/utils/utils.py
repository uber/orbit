import numpy as np
import pandas as pd
import string

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression


def simulate_seasonal_term(periodicity, total_cycles, noise_std=1.,
                           harmonics=None):
    """Generates a seasonality term"""
    # https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_seasonal.html
    duration = periodicity * total_cycles
    assert duration == int(duration)
    duration = int(duration)
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = noise_std * np.random.randn((harmonics))
    gamma_star_jt = noise_std * np.random.randn((harmonics))

    total_timesteps = 100 * duration # Pad for burn in
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (gamma_jt[j - 1] * cos_j
                                 + gamma_star_jt[j - 1] * sin_j
                                 + noise_std * np.random.randn())
            gamma_star_jtp1[j - 1] = (- gamma_jt[j - 1] * sin_j
                                      + gamma_star_jt[j - 1] * cos_j
                                      + noise_std * np.random.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1
    wanted_series = series[-duration:] # Discard burn in

    return wanted_series


def make_synthetic_series(seed=0, date_col='week', periodicity=52, total_cycles=4, harmonics=3, freq='7D'):
    """Generate synthetic data with regressors"""
    np.random.seed(seed)

    # simulate seasonality
    seasonality_term = simulate_seasonal_term(periodicity=periodicity, total_cycles=total_cycles, harmonics=harmonics)

    # scale data
    scaler = MinMaxScaler(feature_range=(np.max(seasonality_term), np.max(seasonality_term) * 2))
    seasonality_term = scaler.fit_transform(seasonality_term[:, None])

    # datetime index
    dt = pd.date_range(start='2016-01-04', periods=len(seasonality_term), freq=freq)

    # create df
    df = pd.DataFrame(seasonality_term, columns=['response'], index=dt).reset_index()
    df = df.rename(columns={'index': date_col})

    # make regression
    X, y, coef = make_regression(n_samples=df.shape[0], n_features=6, n_informative=3, random_state=seed, coef=True)

    scaler = MinMaxScaler(feature_range=(1, np.max(X) * 2))
    X = scaler.fit_transform(X)
    X = X / 2

    n_samples, n_features = X.shape
    feature_names = list(string.ascii_letters[:n_features])
    X_df = pd.DataFrame(X, columns=feature_names)

    # join regression
    df = pd.concat((df, X_df), axis=1)

    # sum response with regression response
    df['response'] = df['response'] + y

    # linear trend
    df['response'] = df['response'] * np.linspace(5, 1, df.shape[0])

    return df, coef
