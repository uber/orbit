import pandas as pd


def load_iclaims():
    """Load iclaims dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://fred.stlouisfed.org/series/ICNSA
    https://trends.google.com/trends/?geo=US
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/iclaims_example.csv'
    df = pd.read_csv(url, parse_dates=['week'])

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


