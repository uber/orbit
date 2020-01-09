import pytest
import pandas as pd


@pytest.fixture
def iclaims_training_data():
    df = pd.read_csv(
        'tests/resources/iclaims.example.csv',
        parse_dates=['week']
    )
    return df
