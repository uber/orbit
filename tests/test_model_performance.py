import pytest
from orbit.lgt import LGT

@pytest.fixture(scope='module')
def lgt_mcmc_mean_with_regression(synthetic_data):
    train_df, test_df, coef = synthetic_data
    lgt = LGT(
        response_col='response',
        date_col='week',
        regressor_col=train_df.columns.tolist()[2:],
        seasonality=52,
        sample_method='mcmc',
        predict_method='mean',
    )
    lgt.fit(train_df)
    return lgt
