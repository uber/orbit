import pandas as pd
import tqdm
from orbit.backtest.backtest import TimeSeriesSplitter, Backtest


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

