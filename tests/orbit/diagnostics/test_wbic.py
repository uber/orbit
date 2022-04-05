import pytest
import numpy as np

from orbit.models import LGT, DLT, KTR, KTRLite


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map"])
def test_wbic_dlt(make_weekly_data, estimator):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=list("abcdef"),
        seasonality=52,
        num_warmup=100,
        num_sample=100,
        seed=6666,
        estimator=estimator,
    )

    if estimator == "stan-mcmc":
        wbic_val = dlt.fit_wbic(df=train_df)
        assert np.isclose(wbic_val, dlt.get_wbic())

    elif estimator == "stan-map":
        dlt.fit(df=train_df)
        dlt.get_bic()


def test_wbic_lgt(make_weekly_data):
    train_df, test_df, coef = make_weekly_data

    lgt1 = LGT(
        response_col="response",
        date_col="week",
        regressor_col=list("abcdef"),
        seasonality=52,
        num_warmup=100,
        num_sample=100,
        seed=6666,
        estimator="stan-mcmc",
    )
    wbic_val1 = lgt1.fit_wbic(df=train_df)

    lgt2 = LGT(
        response_col="response",
        date_col="week",
        regressor_col=list("abcdef"),
        seasonality=52,
        num_steps=100,
        num_sample=100,
        seed=6666,
        estimator="pyro-svi",
    )
    wbic_val2 = lgt2.fit_wbic(df=train_df)

    assert np.abs(wbic_val2 - wbic_val1) / wbic_val1 <= 0.05

    lgt3 = LGT(
        response_col="response",
        date_col="week",
        regressor_col=list("abcdef"),
        seasonality=52,
        seed=6666,
        estimator="stan-map",
    )
    lgt3.fit(df=train_df)
    bic_val = lgt3.get_bic()


def test_wbic_ktr(make_weekly_data):
    train_df, test_df, coef = make_weekly_data

    ktr = KTR(
        response_col="response",
        date_col="week",
        regressor_col=list("abcdef"),
        seasonality=52,
        num_steps=100,
        num_sample=100,
        seed=6666,
        estimator="pyro-svi",
    )

    wbic_val = ktr.fit_wbic(df=train_df)
    assert np.isclose(wbic_val, ktr.get_wbic())


def test_wbic_ktrlite(make_weekly_data):
    train_df, test_df, coef = make_weekly_data

    ktrlite = KTRLite(
        response_col="response",
        date_col="week",
        seasonality=52,
        seed=6666,
        estimator="stan-map",
    )

    ktrlite.fit(df=train_df)
    ktrlite.get_bic()
