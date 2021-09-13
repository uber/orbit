import pandas as pd
from copy import deepcopy
import inspect
import tqdm
from itertools import product

from ..diagnostics.metrics import smape, wmape, mape, mse, mae, rmsse
from ..diagnostics.backtest import BackTester
from collections.abc import Mapping, Iterable


def grid_search_orbit(param_grid, model, df, min_train_len=None,
                      incremental_len=None, forecast_len=None, n_splits=None,
                      metrics=None, criteria=None, verbose=True, **kwargs):
    """A gird search unitlity to tune the hyperparameters for orbit template using the orbit.diagnostics.backtest modules.
    Parameters
    ----------
    param_gird : dict
        a dict with candidate values for hyper-params to be tuned
    model : object
        model object
    df : pd.DataFrame
    min_train_len : int
        scheduling parameter in backtest
    incremental_len : int
        scheduling parameter in backtest
    forecast_len : int
        scheduling parameter in backtest
    n_splits : int
        scheduling parameter in backtest
    metrics : function
        metric function, defaul smape defined in orbit.diagnostics.metrics
    criteria : str
        "min" or "max"; defatul is None ("min")
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

    def _get_params(model):
        init_args_tmpl = dict()
        init_args = dict()

        # get all the signatures in the hierarchy of model templates
        for cls in inspect.getmro(model._model.__class__):
            sig = inspect.signature(cls)
            for key in sig.parameters.keys():
                if key != 'kwargs':
                    if hasattr(model._model, key):
                        init_args_tmpl[key] = getattr(model._model, key)
        # get all the signatures in the hierarchy of forecaster
        for cls in inspect.getmro(model.__class__):
            sig = inspect.signature(cls)
            for key in sig.parameters.keys():
                if key != 'kwargs':
                    if hasattr(model, key):
                        init_args[key] = getattr(model, key)
        # deal with the estimator separately
        for cls in inspect.getmro(model.estimator_type):
            sig = inspect.signature(cls)
            for key in sig.parameters.keys():
                if key != 'kwargs':
                    if hasattr(model.estimator, key):
                        init_args[key] = getattr(model.estimator, key)

        return init_args_tmpl.copy(), init_args.copy()

    def _yield_param_grid(param_grid):
        # an internal function to mimic the ParameterGrid from scikit-learn
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                                'a list ({!r})'.format(param_grid))

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

    param_list_dict = list(_yield_param_grid(param_grid))
    params_tmpl, params = _get_params(model)
    res = pd.DataFrame(param_list_dict)
    metric_values = list()

    for tuned_param_dict in tqdm.tqdm(param_list_dict):
        if verbose:
            print("tuning hyper-params {}".format(tuned_param_dict))

        params_ = params.copy()
        params_tmpl_ = params_tmpl.copy()
        for key, val in tuned_param_dict.items():
            if key in params_tmpl_.keys():
                params_tmpl_[key] = val
            elif key in params_.keys():
                params_[key] = val
            else:
                raise Exception("tuned hyper-param {} is not in the model's parameters".format(key))

        # it is safer to reinstantiate a model object than using deepcopy...
        new_model_template = model._model.__class__(**params_tmpl_)
        new_model = model.__class__(model=new_model_template, **params_)

        bt = BackTester(
            model=new_model,
            df=df,
            min_train_len=min_train_len,
            n_splits=n_splits,
            incremental_len=incremental_len,
            forecast_len=forecast_len,
            **kwargs
        )
        bt.fit_predict()
        # TODO: should we assert len(metrics) == 1?
        if metrics is None:
            metrics = smape
        metric_val = bt.score(metrics=[metrics]).metric_values[0]
        if verbose:
            print("tuning metric:{:-.5g}".format(metric_val))
        metric_values.append(metric_val)
    res['metrics'] = metric_values
    if criteria is None:
        criteria = 'min'
    best_params = res[res['metrics'] == res['metrics'].apply(criteria)].drop('metrics', axis=1).to_dict('records')

    return best_params, res
