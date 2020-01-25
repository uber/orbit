import pandas as pd
import numpy as np
import tqdm
from orbit.backtest.backtest import Backtest


def run_multi_series_backtest(data, response_col, key_col, models,
                              min_train_len, incremental_len, forecast_len,
                              predicted_col='prediction', n_splits=None, model_names=None,
                              model_callbacks=None, fit_callbacks=None, pred_callbacks=None,
                              fit_args=None, scheme='expanding'):
    # store result per series key
    metric_dict = {}
    unique_keys = data[key_col].unique()
    all_result = []
    all_scores = []
    for key in tqdm.tqdm(unique_keys):
        df = data[data[key_col] == key]
        bt = Backtest(df=df, min_train_len=min_train_len, incremental_len=incremental_len,
                      forecast_len=forecast_len, n_splits=n_splits, scheme=scheme
                      )

        bt.fit_score_batch(models=models, response_col=response_col, predicted_col=predicted_col,
                           model_names=model_names, model_callbacks=model_callbacks,
                           fit_callbacks=fit_callbacks, predict_callbacks=pred_callbacks,
                           fit_args=fit_args)

        all_result.append(bt.get_predictions())
        all_scores.append(bt.get_scores())

    all_result = pd.concat(all_result, axis=0, ignore_index=True)
    all_scores = pd.concat(all_scores, axis=0, ignore_index=True)
    return all_result, all_scores
