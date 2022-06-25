import pandas as pd
import inspect
from tqdm.auto import tqdm
from itertools import product
from collections.abc import Mapping, Iterable

from ..diagnostics.metrics import smape
from ..diagnostics.backtest import BackTester
from ..exceptions import IllegalArgument
from ..forecaster import MAPForecaster

import logging

logger = logging.getLogger("orbit")


def grid_search_orbit(
    param_grid,
    model,
    df,
    eval_method="backtest",
    min_train_len=None,
    incremental_len=None,
    forecast_len=None,
    n_splits=None,
    metrics=None,
    criteria="min",
    verbose=False,
    **kwargs,
):
    """A gird search utility to tune the hyperparameters for orbit template using the orbit.diagnostics.backtest modules.
    Parameters
    ----------
    param_grid : dict
        a dict with candidate values for hyper-params to be tuned
    model : object
        model object
    df : pd.DataFrame
    eval_method : str
        "backtest" or "bic"
    min_train_len : int
        scheduling parameter in backtest
    incremental_len : int
        scheduling parameter in backtest
    forecast_len : int
        scheduling parameter in backtest
    n_splits : int
        scheduling parameter in backtest
    metrics : callable
        metric function in use when evel_method is "backtest";
        if not provided, default will be set as smape defined in orbit.diagnostics.metrics
    criteria : str
        "min" or "max"
    verbose : bool

    Return
    ------
    dict:
        best hyperparams
    pd.DataFrame:
        data frame of tuning results

    """
    # def _get_params(model):
    #     # get all the model params for orbit typed template
    #     params = {}
    #     for key, val in model.__dict__.items():
    #         if not key.startswith('_') and key != 'estimator':
    #             params[key] = val

    #     for key, val in model.__dict__['estimator'].__dict__.items():
    #         if not key.startswith('_') and key != 'stan_init':
    #             params[key] = val

    #     return params.copy()
    if eval_method not in ["backtest", "bic"]:
        raise IllegalArgument(
            "Invalid input of eval_method. Argument not in ['backtest', 'bic']"
        )

    if eval_method == "bic" and criteria != "min":
        logger.info("crtieria is enforced to be min when using 'bic' as eval_method.")
        criteria = "min"

    if criteria not in ["min", "max"]:
        raise IllegalArgument(
            "Invalid input of criteria. Argument not in ['min', 'max']"
        )

    def _get_params(model):
        init_args_tmpl = dict()
        init_args = dict()

        # get all the signatures in the hierarchy of model templates
        for cls in inspect.getmro(model._model.__class__):
            sig = inspect.signature(cls)
            for key in sig.parameters.keys():
                if key != "kwargs":
                    if hasattr(model._model, key):
                        init_args_tmpl[key] = getattr(model._model, key)
        # get all the signatures in the hierarchy of forecaster
        for cls in inspect.getmro(model.__class__):
            sig = inspect.signature(cls)
            for key in sig.parameters.keys():
                if key != "kwargs":
                    if hasattr(model, key):
                        init_args[key] = getattr(model, key)
        # deal with the estimator separately
        for cls in inspect.getmro(model.estimator_type):
            sig = inspect.signature(cls)
            for key in sig.parameters.keys():
                if key != "kwargs":
                    if hasattr(model.estimator, key):
                        init_args[key] = getattr(model.estimator, key)

        return init_args_tmpl.copy(), init_args.copy()

    param_list_dict = generate_param_args_list(param_grid)
    params_tmpl, params = _get_params(model)
    res = pd.DataFrame(param_list_dict)
    metric_values = list()

    for tuned_param_dict in tqdm(param_list_dict):
        if verbose:
            logger.info("tuning hyper-params {}".format(tuned_param_dict))

        params_ = params.copy()
        params_tmpl_ = params_tmpl.copy()
        for key, val in tuned_param_dict.items():
            if key in params_tmpl_.keys():
                params_tmpl_[key] = val
            elif key in params_.keys():
                params_[key] = val
            else:
                raise IllegalArgument(
                    "tuned hyper-param {} is not in the model's parameters".format(key)
                )

        # it is safer to re-instantiate a model object than using deepcopy...
        new_model_template = model._model.__class__(**params_tmpl_)
        new_model = model.__class__(model=new_model_template, **params_)

        if eval_method == "backtest":
            bt = BackTester(
                model=new_model,
                df=df,
                min_train_len=min_train_len,
                n_splits=n_splits,
                incremental_len=incremental_len,
                forecast_len=forecast_len,
                **kwargs,
            )
            bt.fit_predict()
            # TODO: should we assert len(metrics) == 1?
            if metrics is None:
                metrics = smape
            metric_val = bt.score(metrics=[metrics]).metric_values[0]
        elif eval_method == "bic":
            if isinstance(new_model, MAPForecaster):
                # pass thru arg for estimator
                new_model.fit(df)
                metric_val = new_model.get_bic()
            else:
                raise IllegalArgument(
                    "eval_method 'bic' only supports 'stan-map' estimator for now."
                )
        if verbose:
            logger.info("tuning metric:{:-.5g}".format(metric_val))
        metric_values.append(metric_val)

    res["metrics"] = metric_values

    best_params = (
        res[res["metrics"] == res["metrics"].apply(criteria)]
        .drop("metrics", axis=1)
        .to_dict("records")
    )

    return best_params, res


def generate_param_args_list(param_grid):
    """An utils similar to sci-kit learn package to generate combinations of args based on spaces of parameters
    provided from users in a dictionary of list format.


    Parameters
    ----------
    param_grid: dict of list
    dictionary where key represents the arg and value represents the list of values proposed for the grid

    Returns
    -------
    list of dict
    the iterated products of all combinations of params generated based on the input param grid
    """

    # an internal function to mimic the ParameterGrid from scikit-learn

    def _yield_param_grid(param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(
                "Parameter grid is not a dict or a list ({!r})".format(param_grid)
            )

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        for p in param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    return list(_yield_param_grid(param_grid))
