import pandas as pd

def load_iclaims():
    """
    Load iclaims dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://fred.stlouisfed.org/series/ICNSA
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/iclaims_example.csv'
    df = pd.read_csv(url)

    return df

def load_m4weekly():
    """
    Load iclaims dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://forecasters.org/resources/time-series-data/m4-competition/
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/m4_weekly.csv'
    df = pd.read_csv(url)

    return df

def load_m5daily():
    """
    Load iclaims dataset

    Returns
    -------
        pandas DataFrame

    Notes
    -----
    https://www.kaggle.com/c/m5-forecasting-accuracy
    """
    url = 'https://raw.githubusercontent.com/uber/orbit/master/examples/data/m5_agg_demand_full.csv'
    df = pd.read_csv(url)

    return df