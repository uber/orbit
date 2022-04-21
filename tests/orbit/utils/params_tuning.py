import pytest

from orbit.models import DLT, LGT
from orbit.utils.params_tuning import grid_search_orbit


@pytest.mark.parametrize(
    "param_grid",
    [
        {
            "level_sm_input": [0.3, 0.5, 0.8],
            "seasonality_sm_input": [0.3, 0.5, 0.8],
        },
        {
            "damped_factor": [0.3, 0.5, 0.8],
            "slope_sm_input": [0.3, 0.5, 0.8],
        },
    ],
)
@pytest.mark.parametrize("eval_method", ["backtest", "bic"])
def test_dlt_grid_tuning(make_weekly_data, param_grid, eval_method):
    train_df, test_df, coef = make_weekly_data
    args = {
        "response_col": "response",
        "date_col": "week",
        "seasonality": 52,
        "estimator": "stan-map",
    }
    dlt = DLT(**args)

    best_params, tuned_df = grid_search_orbit(
        param_grid,
        model=dlt,
        df=train_df,
        eval_method=eval_method,
        min_train_len=80,
        incremental_len=20,
        forecast_len=20,
    )

    assert best_params[0].keys() == param_grid.keys()
    assert set(tuned_df.columns.to_list()) == set(list(param_grid.keys()) + ["metrics"])
    assert tuned_df.shape == (9, 3)


@pytest.mark.parametrize(
    "param_grid",
    [
        {
            "level_sm_input": [0.3, 0.5, 0.8],
            "seasonality_sm_input": [0.3, 0.5, 0.8],
        },
        {
            "level_sm_input": [0.3, 0.5, 0.8],
            "slope_sm_input": [0.3, 0.5, 0.8],
        },
    ],
)
@pytest.mark.parametrize("eval_method", ["backtest", "bic"])
def test_lgt_grid_tuning(make_weekly_data, param_grid, eval_method):
    train_df, test_df, coef = make_weekly_data
    args = {
        "response_col": "response",
        "date_col": "week",
        "seasonality": 52,
        "estimator": "stan-map",
    }
    lgt = LGT(**args)

    best_params, tuned_df = grid_search_orbit(
        param_grid,
        model=lgt,
        df=train_df,
        eval_method=eval_method,
        min_train_len=80,
        incremental_len=20,
        forecast_len=20,
    )

    assert best_params[0].keys() == param_grid.keys()
    assert set(tuned_df.columns.to_list()) == set(list(param_grid.keys()) + ["metrics"])
    assert tuned_df.shape == (9, 3)
