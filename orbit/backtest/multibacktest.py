import pandas as pd
import numpy as np
import tqdm
from orbit.dlt import DLT
from orbit.backtest.backtest import TimeSeriesSplitter, Backtest
from orbit.utils.metrics import mape, smape, wmape
from orbit.exceptions import BacktestException


def run_multi_series_backtest(data, response_col, key_col, model,
                              min_train_len, incremental_len, forecast_len,
                              predicted_col='prediction', date_col=None, n_splits=None,
                              model_callback=None, fit_callback=None, predict_callback=None,
                              fit_args=None, predict_args=None, window_type='expanding'):

    # store result per series key
    # data = data.copy()
    unique_keys = data[key_col].unique()
    all_result = []
    all_scores = []

    for key in tqdm.tqdm(unique_keys):
        df = data[data[key_col] == key]
        splitter = TimeSeriesSplitter(
            df=df, min_train_len=min_train_len, incremental_len=incremental_len,
            forecast_len=forecast_len, n_splits=n_splits, window_type=window_type,
            date_col=date_col,
        )

        bt = Backtest(splitter)
        bt.fit_score(model=model, response_col=response_col, predicted_col=predicted_col,
                     model_callback=model_callback,
                     fit_callback=fit_callback, predict_callback=predict_callback,
                     fit_args=fit_args, predict_args=predict_args)

        all_result.append(bt.get_predictions())
        scores_df = bt.get_scores()
        scores_df[key_col] = key
        all_scores.append(scores_df)

    all_result = pd.concat(all_result, axis=0, ignore_index=True)
    all_scores = pd.concat(all_scores, axis=0, ignore_index=True)
    return all_result, all_scores

def tune_damped_factor(model,
                       splitter,
                       damped_factor_grid,
                       predicted_col = 'prediction',
                       metrics={"smape": smape, "mape": mape}):
    ''' Utility to tune the damped factor in DLT model with the backtest engine in orbit

    Parameters
    ----------
    model : instance of orbit.dlt.DLT
        this is the underlying model to be tuned with the damped factor
    splitter : instance of orbit.backtest.backtest import TimeSeriesSplitter
        this is the time series splitter used to conduct the backtesting
    damped_factor_grid : list or array-like
        a list of values between 0 and 1; the damped factor candidates
    predicted_col : str
        optional callback to adapt data to work with `model`'s `fit()` method
    metrics : dict
        dictionary of functions `f` which score performance by f(actual_array, predicted_array)

    Returns
    -------
    Pandas Data Frame :
        the aggregated backtesting metrics for each damped factor. Users could pick the dampted fator
        with the smallest out-of-time metrics.

    '''

    if not isinstance(model, DLT):
        raise BacktestException('model should be an instance of orbit.dlt.DLT.')

    if not isinstance(splitter, TimeSeriesSplitter):
        raise BacktestException('splitter should be an instance of orbit.backtest.backtest.TimeSeriesSplitter.')

    if not (all(np.array(damped_factor_grid) > 0) and
            all (np.array(damped_factor_grid) < 1)):
        raise BacktestException('Damped factor should be a value between 0 and 1.')

    bt = Backtest(splitter = splitter)
    bt_res = []
    response_col = model.response_col

    for damped_factor in damped_factor_grid:
        print("Doing backtesting for damped factor {}".format(damped_factor))
        params = model.get_params().copy()
        params['damped_factor_fixed'] = damped_factor

        model_ = DLT(**params)
        bt.fit_score(model = model_,
                     response_col = response_col,
                     predicted_col = predicted_col,
                     metrics = metrics,
                     include_steps = False,
                     insample_predict=True)
        # in-time
        score_df_in = bt.get_insample_scores().iloc[:,:-1]
        score_df_in.columns = ['train_' + col for col in score_df_in.columns]
        # out-of-time
        score_df_oot = bt.get_scores().iloc[:,:-1]
        score_df_oot.columns = ['test_' + col for col in score_df_oot.columns]

        bt_res.append(pd.concat([pd.DataFrame({'damped factor': [damped_factor]}),
                                 score_df_in, score_df_oot], axis=1))

    return pd.concat(bt_res, axis=0, ignore_index=True)
