from copy import copy
import pytest
import numpy as np

from orbit.models import DLT
from orbit.constants.constants import PredictionKeys
from orbit.exceptions import ModelException, PredictionException


@pytest.mark.parametrize("estimator", ["stan-map", "stan-mcmc"])
def test_base_dlt_init(estimator):
    dlt = DLT(estimator=estimator)

    is_fitted = dlt.is_fitted()

    model_data_input = dlt.get_training_data_input()
    model_param_names = dlt._model.get_model_param_names()
    init_values = dlt._model.get_init_values()

    # model is not yet fitted
    assert not is_fitted
    # should only be initialized and not set
    assert not model_data_input
    # model param names should already be set
    assert model_param_names
    # callable is not implemented yet
    assert not init_values


@pytest.mark.parametrize("estimator", ["stan-mcmc"])
def test_dlt_full_univariate(make_weekly_data, estimator):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator=estimator,
    )

    dlt.fit(train_df)

    init_values = dlt._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    predict_df = dlt.predict(test_df)

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(dlt._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator", ["stan-mcmc"])
@pytest.mark.parametrize("point_method", ["mean", "median"])
def test_dlt_aggregated_univariate(make_weekly_data, estimator, point_method):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator=estimator,
    )

    dlt.fit(train_df, point_method=point_method)

    init_values = dlt._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    predict_df = dlt.predict(test_df)

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(dlt._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


def test_dlt_map_univariate(make_weekly_data):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator="stan-map",
    )

    dlt.fit(train_df)

    init_values = dlt._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    predict_df = dlt.predict(test_df)

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(dlt._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator", ["stan-mcmc"])
def test_dlt_non_seasonal_fit(make_weekly_data, estimator):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        estimator=estimator,
        num_warmup=50,
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_num_parameters = len(dlt._model.get_model_param_names()) + 1

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize(
    "regressor_signs",
    [
        ["+", "+", "+", "+", "+", "+"],
        ["-", "-", "-", "-", "-", "-"],
        ["=", "=", "=", "=", "=", "="],
    ],
    ids=["positive_only", "negative_only", "regular_only"],
)
@pytest.mark.parametrize(
    "invalid_input",
    [np.nan, np.inf, -1 * np.inf],
    ids=["nan", "infinite", "neg-infinite"],
)
def test_invalid_regressor(make_weekly_data, regressor_signs, invalid_input):
    train_df, test_df, coef = make_weekly_data
    regressor_col = train_df.columns.tolist()[2:]
    # make invalid values
    train_df[regressor_col[0]][36] = invalid_input
    expected_flag = False
    try:
        dlt = DLT(
            response_col="response",
            date_col="week",
            regressor_col=regressor_col,
            regressor_sign=regressor_signs,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_warmup=50,
            verbose=False,
            estimator="stan-map",
        )
        dlt.fit(train_df)
    except ModelException:
        expected_flag = True

    assert expected_flag


@pytest.mark.parametrize(
    "invalid_input",
    [np.nan, np.inf, -1 * np.inf],
    ids=["nan", "infinite", "neg-infinite"],
)
def test_invalid_predict_regressor(make_weekly_data, invalid_input):
    train_df, test_df, coef = make_weekly_data
    regressor_col = train_df.columns.tolist()[2:]
    # make invalid values
    test_df[regressor_col[0]][3] = invalid_input
    expected_flag = False
    try:
        dlt = DLT(
            response_col="response",
            date_col="week",
            regressor_col=regressor_col,
            prediction_percentiles=[5, 95],
            seasonality=52,
            num_warmup=50,
            verbose=False,
            estimator="stan-map",
        )
        dlt.fit(train_df)
        dlt.predict(test_df)
    except PredictionException:
        expected_flag = True

    assert expected_flag


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
def test_dlt_full_with_regression(make_weekly_data, regressor_signs):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator="stan-mcmc",
    )

    dlt.fit(train_df)
    init_values = dlt._model.get_init_values()
    assert init_values["init_sea"].shape == (51,)

    if regressor_signs.count("+") > 0:
        assert init_values["pr_beta"].shape == (regressor_signs.count("+"),)
    if regressor_signs.count("-") > 0:
        assert init_values["nr_beta"].shape == (regressor_signs.count("-"),)
    if regressor_signs.count("=") > 0:
        assert init_values["rr_beta"].shape == (regressor_signs.count("="),)

    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction_5", "prediction", "prediction_95"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 7)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])

    assert np.sum(regression_out["coefficient"].values >= 0) <= regressor_signs.count(
        "+"
    ) + regressor_signs.count("=")
    assert np.sum(regression_out["coefficient"].values <= 0) <= regressor_signs.count(
        "-"
    ) + regressor_signs.count("=")


@pytest.mark.parametrize("estimator", ["stan-mcmc"])
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
def test_dlt_aggregated_with_regression(
    make_weekly_data, estimator, regressor_signs, point_method
):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator=estimator,
    )

    dlt.fit(train_df, point_method=point_method)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 7)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])

    predict_df = dlt.predict(test_df, decompose=True)
    assert any(predict_df["regression"].values)


@pytest.mark.parametrize(
    "global_trend_option", ["linear", "loglinear", "logistic", "flat"]
)
def test_dlt_map_global_trend(make_weekly_data, global_trend_option):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        seasonality=52,
        global_trend_option=global_trend_option,
        estimator="stan-map",
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns


@pytest.mark.parametrize(
    # true signs
    # coef=[0.2, 0.1, 0.3, 0.15, -0.2, -0.1]
    "regressor_signs",
    [
        ["+", "+", "+", "+", "=", "="],
        ["=", "=", "=", "=", "-", "-"],
        ["+", "+", "+", "+", "-", "-"],
        ["+", "+", "+", "=", "=", "-"],
    ],
    ids=["positive_mixed", "negative_mixed", "positive_negative", "mixed"],
)
def test_dlt_mixed_signs_and_order(make_weekly_data, regressor_signs):
    df, _, _ = make_weekly_data
    raw_regressor_col = ["a", "b", "c", "d", "e", "f"]
    new_regressor_col = [raw_regressor_col[idx] for idx in [1, 2, 0, 5, 3, 4]]
    new_regressor_signs = [regressor_signs[idx] for idx in [1, 2, 0, 5, 3, 4]]

    # mixing ordering of cols in df of prediction
    new_df = df[["response", "week"] + new_regressor_col]

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=raw_regressor_col,
        regressor_sign=regressor_signs,
        seasonality=52,
        seed=8888,
        num_warmup=4000,
        num_sample=4000,
        estimator="stan-mcmc",
    )
    dlt.fit(df)
    coef_df = dlt.get_regression_coefs().set_index("regressor")
    coefs = coef_df.loc[raw_regressor_col, "coefficient"].values

    predicted_df_v1 = dlt.predict(df, decompose=True)
    predicted_df_v2 = dlt.predict(new_df, decompose=True)

    # mixing ordering of signs
    dlt_new = DLT(
        response_col="response",
        date_col="week",
        regressor_col=new_regressor_col,
        regressor_sign=new_regressor_signs,
        seasonality=52,
        seed=8888,
        num_warmup=4000,
        num_sample=4000,
        estimator="stan-mcmc",
    )
    dlt_new.fit(df)
    new_coef_df = dlt_new.get_regression_coefs().set_index("regressor")
    new_coefs = new_coef_df.loc[raw_regressor_col, "coefficient"].values

    # coefficients should be as close as  <= 0.01
    assert np.allclose(coefs, new_coefs, atol=1e-2)

    predicted_df_v3 = dlt_new.predict(df, decompose=True)
    predicted_df_v4 = dlt_new.predict(new_df, decompose=True)

    # relative ratio of regression comp to acutal
    pred_v1 = predicted_df_v1["regression"].values / df["response"].values
    pred_v2 = predicted_df_v2["regression"].values / df["response"].values
    pred_v3 = predicted_df_v3["regression"].values / df["response"].values
    pred_v4 = predicted_df_v4["regression"].values / df["response"].values

    # they should be all identical; ordering of signs or columns in prediction df should show not material difference
    # exclude the first one which is used for initialization and hence more unstable
    # exchange columns in prediction should have minimal variance
    assert np.allclose(pred_v1[1:], pred_v2[1:], atol=1e-3)
    assert np.allclose(pred_v3[1:], pred_v4[1:], atol=1e-3)
    # exchange columns in input of fitting should have less variance
    assert np.allclose(pred_v1[1:], pred_v3[1:], atol=1e-1)


@pytest.mark.parametrize("prediction_percentiles", [None, [5, 10, 95]])
def test_dlt_prediction_percentiles(iclaims_training_data, prediction_percentiles):
    df = iclaims_training_data

    dlt = DLT(
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

    dlt.fit(df)
    predicted_df = dlt.predict(df)
    expected_columns = ["week"] + ["prediction" + p for p in p_labels]
    assert predicted_df.columns.tolist() == expected_columns
    assert predicted_df.shape[0] == df.shape[0]

    predicted_df = dlt.predict(df, decompose=True)
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
        ["-", "-", "-", "-", "-", "-"],
        ["+", "=", "+", "=", "+", "+"],
        ["-", "=", "-", "=", "-", "="],
        ["+", "=", "+", "=", "-", "-"],
    ],
    ids=[
        "positive_only",
        "regular_only",
        "negative_only",
        "positive_mixed",
        "negative_mixed",
        "mixed_signs",
    ],
)
@pytest.mark.parametrize("seasonality", [1, 52])
def test_dlt_full_reproducibility(
    make_weekly_data, estimator, regressor_signs, seasonality
):
    train_df, test_df, coef = make_weekly_data

    dlt_first = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator=estimator,
    )

    # first fit and predict
    dlt_first.fit(train_df)
    posteriors_first = copy(dlt_first._posterior_samples)
    predict_df_first = dlt_first.predict(test_df)
    regression_out_first = dlt_first.get_regression_coefs()

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    dlt_second = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regressor_sign=regressor_signs,
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        num_warmup=50,
        verbose=False,
        estimator=estimator,
    )

    dlt_second.fit(train_df)
    posteriors_second = copy(dlt_second._posterior_samples)
    predict_df_second = dlt_second.predict(test_df)
    regression_out_second = dlt_second.get_regression_coefs()

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
def test_dlt_map_reproducibility(make_weekly_data, seasonality):
    train_df, test_df, coef = make_weekly_data

    dlt1 = DLT(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        estimator="stan-map",
    )

    # first fit and predict
    dlt1.fit(train_df)
    posteriors1 = copy(dlt1._point_posteriors["map"])
    prediction1 = dlt1.predict(test_df)

    # second fit and predict
    # note a new instance must be created to reset the seed
    # note both fit and predict contain random generation processes
    dlt2 = DLT(
        response_col="response",
        date_col="week",
        prediction_percentiles=[5, 95],
        seasonality=seasonality,
        estimator="stan-map",
    )

    dlt2.fit(train_df)
    posteriors2 = copy(dlt2._point_posteriors["map"])
    prediction2 = dlt2.predict(test_df)

    # assert same posterior keys
    assert set(posteriors1.keys()) == set(posteriors2.keys())

    # assert posterior draws are reproducible
    for k, v in posteriors1.items():
        assert np.allclose(posteriors1[k], posteriors2[k])

    # assert prediction is reproducible
    assert np.allclose(
        prediction1["prediction"].values, prediction2["prediction"].values
    )


@pytest.mark.parametrize("regression_penalty", ["fixed_ridge", "lasso", "auto_ridge"])
def test_dlt_regression_penalty(make_weekly_data, regression_penalty):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        regression_penalty=regression_penalty,
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator="stan-map",
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("level_sm_input", [0.0001, 0.5, 1.0])
@pytest.mark.parametrize("seasonality_sm_input", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("slope_sm_input", [0.0, 0.5, 1.0])
def test_dlt_fixed_sm_input(
    make_weekly_data, level_sm_input, seasonality_sm_input, slope_sm_input
):
    train_df, test_df, coef = make_weekly_data

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        level_sm_input=level_sm_input,
        seasonality_sm_input=seasonality_sm_input,
        slope_sm_input=slope_sm_input,
        seasonality=52,
        num_warmup=50,
        verbose=False,
        estimator="stan-map",
    )

    dlt.fit(train_df)
    predict_df = dlt.predict(test_df)

    regression_out = dlt.get_regression_coefs()
    num_regressors = regression_out.shape[0]

    expected_columns = ["week", "prediction"]
    expected_shape = (51, len(expected_columns))
    expected_regression_shape = (6, 3)

    assert predict_df.shape == expected_shape
    assert predict_df.columns.tolist() == expected_columns
    assert regression_out.shape == expected_regression_shape
    assert num_regressors == len(train_df.columns.tolist()[2:])


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map"])
def test_dlt_missing(iclaims_training_data, estimator):
    df = iclaims_training_data
    missing_idx = np.array([10, 20, 30, 40, 41, 42, 43, 44, df.shape[0] - 1])
    df.loc[missing_idx, "claims"] = np.nan

    dlt = DLT(
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


def test_dlt_map_single_regressor(iclaims_training_data):
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])
    regressor_col = ["trend.unemploy"]

    dlt = DLT(
        response_col="claims",
        date_col="week",
        regressor_col=regressor_col,
        seasonality=52,
        seed=8888,
        estimator="stan-map",
    )
    dlt.fit(df)
    predicted_df = dlt.predict(df)

    expected_num_parameters = len(dlt._model.get_model_param_names()) + 1
    expected_columns = ["week", "prediction"]

    assert predicted_df.shape[0] == df.shape[0]
    assert predicted_df.columns.tolist() == expected_columns
    assert len(dlt._posterior_samples) == expected_num_parameters


@pytest.mark.parametrize("estimator", ["stan-mcmc"])
@pytest.mark.parametrize("keep_samples", [True, False])
@pytest.mark.parametrize("point_method", ["mean", "median"])
def test_dlt_is_fitted(iclaims_training_data, estimator, keep_samples, point_method):
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])
    regressor_col = ["trend.unemploy"]

    dlt = DLT(
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

    dlt.fit(df, keep_samples=keep_samples, point_method=point_method)
    is_fitted = dlt.is_fitted()

    # still True when keep_samples is False
    assert is_fitted


@pytest.mark.parametrize("estimator", ["stan-mcmc", "stan-map"])
@pytest.mark.parametrize("random_seed", [10, 100])
def test_dlt_predict_seed(make_weekly_data, estimator, random_seed):
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

    dlt = DLT(**args)
    dlt.fit(train_df)
    predict_df1 = dlt.predict(test_df, seed=random_seed)
    predict_df2 = dlt.predict(test_df, seed=random_seed)

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
def test_dlt_predict_range(make_weekly_data, idx_range):
    train_cut_off = 100
    base_df, _, _ = make_weekly_data
    train_df = base_df[:train_cut_off].reset_index(drop=True)
    predict_df = base_df[idx_range[0] : idx_range[1]].reset_index(drop=True)

    dlt = DLT(
        response_col="response",
        date_col="week",
        regressor_col=train_df.columns.tolist()[2:],
        seasonality=52,
        verbose=False,
        estimator="stan-map",
    )
    dlt.fit(train_df)
    dlt.predict(predict_df)
