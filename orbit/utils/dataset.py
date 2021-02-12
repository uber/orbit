import pandas as pd
import numpy as np


def load_iclaims(end_date='2018-06-24'):
    """Load iclaims dataset

    Returns
    -------
        pd.DataFrame

    Notes
    -----
    iclaims is a dataset containing the weekly initial claims for US unemployment benefits against a few related google
    trend queries (unemploy, filling and job)from Jan 2010 - June 2018. This aims to mimick the dataset from the paper
    Predicting the Present with Bayesian Structural Time Series by SCOTT and VARIAN (2014).
    Number of claims are obtained from [Federal Reserve Bank of St. Louis] while google queries are obtained through
    Google Trends API.

    Note that dataset is transformed by natural log before fitting in order to be fitted as a multiplicative model.

    https://fred.stlouisfed.org/series/ICNSA
    https://trends.google.com/trends/?geo=US
    https://finance.yahoo.com/
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/iclaims_example.csv'
    df = pd.read_csv(url, parse_dates=['week'])
    df = df[df['week'] <= end_date]

    # standardize the regressors by mean; equivalent to subtracting mean after np.log
    regressors = ['trend.unemploy', 'trend.filling', 'trend.job', 'sp500', 'vix']

    # convert to float
    for col in regressors:
        df[col] = df[col].astype(float)

    # log transfer
    df[['claims'] + regressors] = df[['claims'] + regressors].apply(np.log)
    # de-mean
    df[regressors] = df[regressors] - df[regressors].apply(np.mean)

    return df


def load_m4weekly():
    """Load m4 weekly sample dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://forecasters.org/resources/time-series-data/m4-competition/
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/m4_weekly.csv'
    df = pd.read_csv(url, parse_dates=['date'])

    return df


def load_m5daily():
    """Load m5 aggregated daily dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://www.kaggle.com/c/m5-forecasting-accuracy
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/m5_agg_demand_full.csv'
    df = pd.read_csv(url, parse_dates=['date'])

    return df


def load_m3monthly():
    """Load m3 monthly sample dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://forecasters.org/resources/time-series-data/m3-competition/
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/m3_monthly.csv'
    df = pd.read_csv(url, parse_dates=['date'])

    return df


def load_electricity_demand():
    """Load Turkish electricity demand daily data from 1 January 2000 to 31 December 2008.
    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://robjhyndman.com/publications/complex-seasonality/

    :return:
    """

    url = 'https://robjhyndman.com/data/turkey_elec.csv'
    df = pd.read_csv(url, header=None, names=['electricity'])
    df['date'] = pd.date_range(start='1/1/2000', end='31/12/2008', freq='D')
    # re-arrange columns
    df = df[['date', 'electricity']]

    return df


def load_air_passengers():
    """Load air passengers csv from prophet github
    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://github.com/facebook/prophet/tree/master/examples

    :return:
    """

    url = 'https://raw.githubusercontent.com/facebook/prophet/master/examples/example_air_passengers.csv'
    df = pd.read_csv(url, parse_dates=['ds'])

    return df
