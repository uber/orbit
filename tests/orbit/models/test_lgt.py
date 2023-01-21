from copy import copy
import pytest
import numpy as np

from orbit.models import LGT
from orbit.constants.constants import PredictionKeys


@pytest.mark.parametrize("estimator", ["stan-map", "stan-mcmc"])
def test_base_lgt_init(estimator):
    lgt = LGT(estimator=estimator)

    is_fitted = lgt.is_fitted()

    model_data_input = lgt.get_training_data_input()
    model_param_names = lgt._model.get_model_param_names()
    init_values = lgt._model.get_init_values()

    # model is not yet fitted
    assert not is_fitted
    # should only be initialized and not set
    assert not model_data_input
    # model param names should already be set
    assert model_param_names
    # callable is not implemented yet
    assert not init_values


@pytest.mark.parametrize("seasonality", [None, 52])
@pytest.mark.parametrize("store_prediction_array", [True, False])
@pytest.mark.parametrize("n_bootstrap_draws", [None, 50])
@pytest.mark.parametrize("point_method", [None, "median"])
@pytest.mark.parametrize("estimator", ["stan-mcmc", "pyro-svi"])
def test_lgt_full_fit(
    make_weekly_data,
    seasonality,
    estimator,
    n_bootstrap_draws,
    store_prediction_array,
    point_method,
):
    train_df, test_df, coef = make_weekly_data
    args = {
        "response_col": "response",
        "date_col": "week",
        "prediction_percentiles": [5, 95],
        "seasonality": seasonality,
        "n_bootstrap_draws": n_bootstrap_draws,
        "verbose": False,
        "estimator": estimator,
    }

    if estimator == "stan-mcmc":
        args.update({"num_warmup": 50, "num_sample": 100})
    elif estimator == "pyro-svi":
        args.update({"num_steps": 10})

    # if seasonality == 52:
    #     expected_num_parameters += 2

    lgt = LGT(**args)
    lgt.fit(train_df, point_method=point_method)
    if estimator == "stan-mcmc":
        expected_num_parameters = len(lgt._model.get_model_param_names()) + 1
    elif estimator == "pyro-svi":
        expected_num_parameters = len(lgt._model.get_model_param_names())

    init_values = lgt._model.get_init_values()
    if seasonality:
        assert init_values["init_sea"].shape == (51,)
    else:
        assert init_values is None

    predict_df = lgt.predict(test_df, store_prediction_array=store_prediction_array)

    _ = lgt.get_prediction_meta()
    _ = lgt.get_training_metrics()

    if not n_bootstrap_draws and point_method:
        expected_columns = ["week", "prediction"]
        expected_shape = (51, len(expected_columns))
    else:
        expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
        expected_shape = (51, len(expected_columns))

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("seasonality", [None, 52])
@pytest.mark.parametrize("estimator", ["stan-mcmc", "pyro-svi"])
@pytest.mark.parametrize("point_method", ["mean", "median"])
def test_lgt_aggregated_fit(make_weekly_data, seasonality, estimator, point_method):
    train_df, test_df, coef = make_weekly_data
    args = {
        "response_col": "response",
        "date_col": "week",
        "prediction_percentiles": [5, 95],
        "seasonality": seasonality,
        "verbose": False,
        "estimator": estimator,
    }
    if estimator == "stan-mcmc":
        args.update({"num_warmup": 50, "num_sample": 50})
    elif estimator == "pyro-svi":
        args.update({"num_steps": 10, "num_sample": 50})

    lgt = LGT(**args)
    lgt.fit(train_df, point_method=point_method)
    if estimator == "stan-mcmc":
        expected_num_parameters = len(lgt._model.get_model_param_names()) + 1
    elif estimator == "pyro-svi":
        expected_num_parameters = len(lgt._model.get_model_param_names())

    init_values = lgt._model.get_init_values()
    if seasonality:
        assert init_values["init_sea"].shape == (51,)
    else:
        assert init_values is None

    predict_df = lgt.predict(test_df)
    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("seasonality", [None, 52])
@pytest.mark.parametrize("estimator", ["stan-map"])
def test_lgt_map_fit(make_weekly_data, seasonality, estimator):
    train_df, test_df, coef = make_weekly_data

    lgt = LGT(
        response_col="response",
        date_col="week",
        seasonality=seasonality,
        verbose=False,
        estimator=estimator,
    )

    lgt.fit(train_df)
    init_values = lgt._model.get_init_values()
    if seasonality:
        assert init_values["init_sea"].shape == (51,)
    else:
        assert init_values is None

    predict_df = lgt.predict(test_df)

    expected_num_parameters = len(lgt._model.get_model_param_names()) + 1
    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator", ["stan-mcmc", "pyro-svi"])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["-", "-", "-", "-", "-", "-"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "-", "-"],
    ],
    ids=["positive_only", "negative_only", "regular_only", "mixed_signs"],
)
def test_lgt_full_with_regression(make_weekly_data, estimator, regressor_signs):
    train_df, test_df, coef = make_weekly_data

    if estimator == "stan-mcmc":
        lgt = LGT(
            response_col="response",
            date_col="week",
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_warmup=50,
            num_sample=50,
            verbose=False,
            estimator=estimator,
        )
    elif estimator == "pyro-svi":
        lgt = LGT(
            response_col="response",
            date_col="week",
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_steps=10,
            verbose=False,
            estimator=estimator,
        )
    else:
        return None

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 7)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator", ["stan-mcmc", "pyro-svi"])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["-", "-", "-", "-", "-", "-"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "-", "-"],
    ],
    ids=["positive_only", "negative_only", "regular_only", "mixed_signs"],
)
@pytest.mark.parametrize("point_method", ["mean", "median"])
def test_lgt_aggregated_with_regression(
    make_weekly_data, estimator, regressor_signs, point_method
):
    train_df, test_df, coef = make_weekly_data

    if estimator == "stan-mcmc":
        lgt = LGT(
            response_col="response",
            date_col="week",
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            num_warmup=50,
            num_sample=50,
            verbose=False,
            estimator=estimator,
        )
    elif estimator == "pyro-svi":
        lgt = LGT(
            response_col="response",
            date_col="week",
            regressor_col=train_df.columns.tolist()[2:],
            regressor_sign=regressor_signs,
            seasonality=52,
            num_steps=10,
            verbose=False,
            estimator=estimator,
        )
    else:
        return None

    lgt.fit(train_df, point_method=point_method)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 7)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])

    predict_df = lgt.predict(test_df, decompose=True)
    assert any(predict_df["regression"].values)


@pytest.mark.parametrize("regressor_signs", [["=", "=", "+"]], ids=["positive_mixed"])
def test_lgt_mixed_signs_and_order(iclaims_training_data, regressor_signs):
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])
    raw_regressor_col = ["trend.unemploy", "trend.filling", "trend.job"]
    new_regressor_col = [raw_regressor_col[idx] for idx in [2, 1, 0]]
    new_regressor_signs = [regressor_signs[idx] for idx in [2, 1, 0]]
    # mixing ordering of cols in df of prediction
    new_df = df[["claims", "week"] + new_regressor_col]

    lgt = LGT(
        response_col="claims",
        date_col="week",
        regressor_col=raw_regressor_col,
        regressor_sign=regressor_signs,
        estimator="stan-map",
        seasonality=52,
        seed=8888,
    )
    lgt.fit(df)
    predicted_df_v1 = lgt.predict(df)
    predicted_df_v2 = lgt.predict(new_df)

    # mixing ordering of signs
    lgt_new = LGT(
        response_col="claims",
        date_col="week",
        regressor_col=new_regressor_col,
        regressor_sign=new_regressor_signs,
        estimator="stan-map",
        seasonality=52,
        seed=8888,
    )
    lgt_new.fit(df)
    predicted_df_v3 = lgt_new.predict(df)
    predicted_df_v4 = lgt_new.predict(new_df)

    pred_v1 = predicted_df_v1["prediction"].values
    pred_v2 = predicted_df_v2["prediction"].values
    pred_v3 = predicted_df_v3["prediction"].values
    pred_v4 = predicted_df_v4["prediction"].values

    # they should be all identical; ordering of signs or columns in prediction show not matter
    assert np.allclose(pred_v1, pred_v2, atol=1e-2)
    assert np.allclose(pred_v1, pred_v3, atol=1e-2)
    assert np.allclose(pred_v1, pred_v4, atol=1e-2)


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_lgt_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    lgt = LGT(
        response_col="claims",
        date_col="week",
        seasonality=52,
        num_warmup=50,
        num_sample=50,
        seed=8888,
        prediction_percentiles=prediction_percentiles,
        estimator="stan-mcmc",
    )

    if not prediction_percentiles:
        p_labels = ["_5", "", "_95"]
    else:
        p_labels = ["_5", "_10", "", "_95"]

    lgt.fit(df)
    predicted_df = lgt.predict(df)
    expected_columns = ["week"] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    predicted_df = lgt.predict(df, decompose=True)
    predicted_components = [
        "prediction",
        PredictionKeys.TREND.value,
        PredictionKeys.SEASONALITY.value,
        PredictionKeys.REGRESSION.value,
    ]

    expected_columns = ["week"]
    for pc in predicted_components:
        for p in p_labels:
            expected_columns.append(pc + p)
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


@pytest.mark.parametrize("estimator", ["stan-mcmc"])
@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["=", "=", "=", "=", "=", "="],
        ["+", "=", "+", "=", "+", "+"],
    ],
    ids=["positive_only", "regular_only", "mixed_signs"],
)
@pytest.mark.parametrize("seasonality", [1, 52])
def test_lgt_full_reproducibility(
    make_weekly_data, estimator, regressor_signs, seasonality
):
    train_df, test_df, coef = make_weekly_data

    lgt_first = LGT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator=estimator,
    )

    # first fit and predict
    lgt_first.fit(train_df)
    posteriors_first = copy(lgt_first._posterior_samples)
    predict_df_first = lgt_first.predict(test_df)
    regression_out_first = lgt_first.get_regression_coefs()

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    lgt_second = LGT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        num_sample=50,
        verbose=False,
        estimator=estimator,
    )

    lgt_second.fit(train_df)
    posteriors_second = copy(lgt_second._posterior_samples)
    predict_df_second = lgt_second.predict(test_df)
    regression_out_second = lgt_second.get_regression_coefs()

    # assert same posterior keys
    assert set(posteriors_first.keys()) == set(posteriors_second.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors_first.items():
        assert np.allclose(posteriors_first[k], posteriors_second[k])

    # assert identical regression columns
    # this is also checked in posterior samples, but an extra layer just in case
    # since this one very commonly retrieved by end users
    assert regression_out_first.equals(regression_out_second)

    # assert prediction is reproducible
    assert predict_df_first.equals(predict_df_second)


@pytest.mark.parametrize("seasonality", [1, 52])
def test_lgt_map_reproducibility(make_weekly_data, seasonality):
    train_df, test_df, coef = make_weekly_data

    lgt1 = LGT(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        estimator="stan-map",
    )

    # first fit and predict
    lgt1.fit(train_df)
    posteriors1 = copy(lgt1._point_posteriors["map"])
    prediction1 = lgt1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    lgt2 = LGT(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        estimator="stan-map",
    )

    lgt2.fit(train_df)
    posteriors2 = copy(lgt2._point_posteriors["map"])
    prediction2 = lgt2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert np.allclose(
        prediction1["prediction"].values, prediction2["prediction"].values
    )


@pytest.mark.parametrize("level_sm_input", [0.0001, 0.5, 1.0])
@pytest.mark.parametrize("seasonality_sm_input", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("slope_sm_input", [0.0, 0.5, 1.0])
def test_lgt_fixed_sm_input(
    make_weekly_data, level_sm_input, seasonality_sm_input, slope_sm_input
):
    train_df, test_df, coef = make_weekly_data

    lgt = LGT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        level_sm_input=level_sm_input,
        seasonality_sm_input=seasonality_sm_input,
        slope_sm_input=slope_sm_input,
        estimator="stan-map",
        seasonality=52,
        verbose=False,
    )

    lgt.fit(train_df)
    predict_df = lgt.predict(test_df)

    regression_out = lgt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map"])
def test_lgt_missing(iclaims_training_data, estimator):
    df = iclaims_training_data
    missing_idx = np.array([10, 20, 30, 40, 41, 42, 43, 44, df.shape[0] - 1])
    df.loc[missing_idx, "claims"] = np.nan

    dlt = LGT(
        response_col="claims",
        date_col="week",
        seasonality=52,
        verbose=False,
        estimator=estimator,
    )

    dlt.fit(df)
    predicted_df = dlt.predict(df)
    if estimator == "stan-map":
        expected_columns = ["week", "prediction"]
    elif estimator == "stan-mcmc":
        expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]

    assert all(~np.isnan(predicted_df["prediction"]))
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]


def test_lgt_map_single_regressor(iclaims_training_data):
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])
    regressor_col = ["trend.unemploy"]

    lgt = LGT(
        response_col="claims",
        date_col="week",
        regressor_col=regressor_col,
        estimator="stan-map",
        seasonality=52,
        seed=8888,
    )
    lgt.fit(df)
    predicted_df = lgt.predict(df)

    expected_num_parameters = len(lgt._model.get_model_param_names()) + 1
    expected_columns = ["week", "prediction"]

    assert predicted_df.shape[0] == df.shape[0]
    assert predicted_df.columns.tolist() == expected_columns
    assert len(lgt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator", ["stan-mcmc", "pyro-svi"])
@pytest.mark.parametrize("keep_samples", [True, False])
@pytest.mark.parametrize("point_method", ["mean", "median"])
def test_lgt_is_fitted(iclaims_training_data, estimator, keep_samples, point_method):
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])
    regressor_col = ["trend.unemploy"]

    if estimator == "stan-mcmc":
        lgt = LGT(
            response_col="claims",
            date_col="week",
            regressor_col=regressor_col,
            seasonality=52,
            seed=8888,
            num_warmup=50,
            num_sample=50,
            verbose=False,
            estimator=estimator,
        )
    elif estimator == "pyro-svi":
        lgt = LGT(
            response_col="claims",
            date_col="week",
            regressor_col=regressor_col,
            seasonality=52,
            seed=8888,
            num_steps=10,
            verbose=False,
            estimator=estimator,
        )
    lgt.fit(df, keep_samples=keep_samples, point_method=point_method)
    is_fitted = lgt.is_fitted()

    # still True when keep_samples is False
    assert is_fitted


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map", "pyro-svi"])
@pytest.mark.parametrize("random_seed", [10, 100])
def test_lgt_predict_seed(make_weekly_data, estimator, random_seed):
    train_df, test_df, coef = make_weekly_data
    args = {
        "response_col": "response",
        "date_col": "week",
        "seasonality": 52,
        "n_bootstrap_draws": 100,
        "verbose": False,
        "estimator": estimator,
    }

    if estimator == "stan-mcmc":
        args.update({"num_warmup": 50, "num_sample": 100})
    elif estimator == "pyro-svi":
        args.update({"num_steps": 10})

    lgt = LGT(**args)
    lgt.fit(train_df)
    predict_df1 = lgt.predict(test_df, seed=random_seed)
    predict_df2 = lgt.predict(test_df, seed=random_seed)

    assert all(predict_df1["prediction"].values == predict_df2["prediction"].values)


@pytest.mark.parametrize(
    "idx_range",
    [
        [0, 100],
        [0, 50],
        [50, 100],
        [50, 80],
        [50, 120],
        [100, 150],
    ],
    ids=[
        "train-start-to-train-end",
        "train-start-to-middle",
        "middle-to-train-end",
        "train-period-subset",
        "train-test-period-cross",
        "completely-test-period",
    ],
)
def test_lgt_predict_range(make_weekly_data, idx_range):
    train_cut_off = 100
    base_df, _, _ = make_weekly_data
    train_df = base_df[:train_cut_off].reset_index(drop=True)
    predict_df = base_df[idx_range[0] : idx_range[1]].reset_index(drop=True)

    lgt = LGT(
        response_col="response",
        date_col="week",
        seasonality=52,
        verbose=False,
        estimator="stan-map",
    )
    lgt.fit(train_df)
    lgt.predict(predict_df)
