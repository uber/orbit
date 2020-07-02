import pandas as pd
import numpy as np

from orbit.constants.backtest import BacktestNames


def to_df(outcome, date_col, response_col):
    ''' Convert a dict structure prediction outcomes to pd.DataFrame
    '''
    # outcome = {
    #     "train_actual": train_df[model.response_col].values,
    #     "train_pred": None,
    #     "train_dts": train_df[model.date_col].values,
    #     "test_pred": predicted_df[self.predicted_col].values,
    #     "test_actual": test_df[model.response_col].values,
    #     "test_dts": test_df[model.date_col].values,
    #     "train_end_dt": None,
    # }

    df = pd.DataFrame({
        BacktestNames.FORECAST_STEPS.value:\
            list(range(len(outcome['train_actual']))) + list(range(len(outcome['test_actual']))),
        date_col: np.concatenate([outcome['train_dts'], outcome['test_dts']]),
        response_col:  np.concatenate([outcome['train_actual'], outcome['test_actual']]),
        BacktestNames.PREDICTED_COL.value: np.nan,
        BacktestNames.TRAIN_TEST_PARTITION.value: ['train'] * outcome['n_train'] + ['test'] * outcome['n_test']
    })

    df[BacktestNames.FORECAST_STEPS.value] += 1
    df.loc[df[BacktestNames.TRAIN_TEST_PARTITION.value] == 'train', BacktestNames.PREDICTED_COL.value] = outcome['train_pred']
    df.loc[df[BacktestNames.TRAIN_TEST_PARTITION.value] == 'test', BacktestNames.PREDICTED_COL.value] = outcome['test_pred']

    return df