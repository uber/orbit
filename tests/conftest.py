import pytest
import pandas as pd
import pkg_resources

from orbit.utils.utils import make_synthetic_series


@pytest.fixture
def iclaims_training_data():
    test_file = pkg_resources.resource_filename(
        'tests',
        'resources/iclaims.example.csv'
    )
    df = pd.read_csv(
        test_file,
        parse_dates=['week']
    )
    return df


@pytest.fixture
def m3_monthly_data():
    test_file = pkg_resources.resource_filename(
        'tests',
        'resources/m3-monthly.csv'
    )
    df = pd.read_csv(
        test_file,
        parse_dates=['date']
    )
    return df


@pytest.fixture
def synthetic_data():
    df, coef = make_synthetic_series(seed=127)

    train_df = df[df['week'] <= '2019-01-01']
    test_df = df[df['week'] > '2019-01-01']

    return train_df, test_df, coef


@pytest.fixture
def valid_sample_predict_method_combo():
    valid_permutations = [
        ("map", "map"),
        ("vi", "mean"), ("vi", "median"), ("vi", "full"),
        ("mcmc", "mean"), ("mcmc", "median"), ("mcmc", "full")
    ]
    return valid_permutations


@pytest.fixture
def valid_pyro_sample_predict_method_combo():
    valid_permutations = [
        ("map", "map"),
        ("vi", "mean"), ("vi", "median"), ("vi", "full")
    ]
    return valid_permutations
