from enum import Enum


class BacktestNames(Enum):
    """ hash table keys for the dictionary of back-test meta data
    """
    # splitting
    MODEL = 'model'
    TRAIN_START_DATE = 'train_start_dt'
    TRAIN_END_DATE = 'train_end_dt'
    # FORECAST_DATE = 'forecast_dt'
    FORECAST_STEPS = 'steps'
    TRAIN_IDX = 'train_idx'
    TEST_IDX = 'test_idx'
    SPLIT_KEY = 'split_key'
    # backtest
    TRAIN_TEST_PARTITION = 'part' # train or # predict
    ACTUALCOL = 'actual'
    PREDICTED_COL = 'prediction'
    # analysis
    METRIC_NAME = 'metric_name'
    METRIC_PER_BTMOD = 'metric_per_btmod'
    METRIC_GEO = 'metric_geo' # FIXME why do we have geo here?
    METRIC_PER_HORIZON = 'metric_per_horizon'