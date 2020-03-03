import pytest
import pandas as pd
import pkg_resources


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
