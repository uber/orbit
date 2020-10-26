import pytest

from orbit.utils.stan import estimate_level_smoothing


@pytest.mark.parametrize("seasonality", [1, 52])
@pytest.mark.parametrize("horizon", [None, 52])
def test_estimate_level_smoothing(synthetic_data, seasonality, horizon):
    train_df, test_df, coef = synthetic_data
    x = train_df['response'].values
    est_lev_sm = estimate_level_smoothing(x, seasonality=seasonality, horizon=horizon)
    assert est_lev_sm >= 1e-5
    assert est_lev_sm <= 1 - 1e-5
