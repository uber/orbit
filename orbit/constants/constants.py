from collections import namedtuple
from enum import Enum


class PredictMethod(Enum):
    """
    The predict method for all of the stan template. Often used are mean and median.
    """
    MAP = 'map'
    MEAN = 'mean'
    MEDIAN = 'median'
    FULL_SAMPLING = 'full'


class PredictionKeys(Enum):
    """
    column names for the data frame of predicted result with decomposed components
    """
    TREND = 'trend'
    SEASONALITY = 'seasonality'
    REGRESSION = 'regression'
    REGRESSOR = 'regressor'
    PREDICTION = 'prediction'


class SupportedEstimators(Enum):
    PyroSVI = 'pyro-svi'
    StanMAP = 'stan-map'
    StanMCMC = 'stan-mcmc'


class PlotLabels(Enum):
    """
    used in multiple prediction plots
    """
    # Also used in training_actual_response column name.
    TRAINING_ACTUAL_RESPONSE = 'training_actual_response'
    PREDICTED_RESPONSE = 'predicted_response'
    ACTUAL_RESPONSE = 'actual_response'


class TimeSeriesSplitSchemeNames(Enum):
    """ hash table keys for the dictionary of back-test meta data
    """
    MODEL = 'model'
    TRAIN_START_DATE = 'train_start_date'
    TRAIN_END_DATE = 'train_end_date'
    TRAIN_IDX = 'train_idx'
    TEST_IDX = 'test_idx'


class BacktestFitColumnNames(Enum):
    """ column names for the data frame of back-test fitting result
    """
    TRAIN_START_DATE = 'train_start_date'
    TRAIN_END_DATE = 'train_end_date'
    FORECAST_DATES = 'forecast_dates'
    ACTUAL = 'actual'
    PREDICTED = 'pred'
    PREDICT_HORIZON = 'pred_horizon'


class BacktestAnalyzeKeys(Enum):
    """ hash table keys for the dictionary of back-test aggregation analysis result
    """
    METRIC_NAME = 'metric_name'
    METRIC_PER_BTMOD = 'metric_per_btmod'
    METRIC_GEO = 'metric_geo'
    METRIC_PER_HORIZON = 'metric_per_horizon'


class CoefPriorDictKeys(Enum):
    """ hash table keys for the dictionary of back-test aggregation analysis result
    """
    NAME = 'name'
    PRIOR_START_TP_IDX = 'prior_start_tp_idx'
    PRIOR_END_TP_IDX = 'prior_end_tp_idx'
    PRIOR_MEAN = 'prior_mean'
    PRIOR_SD = 'prior_sd'
    PRIOR_REGRESSOR_COL = 'prior_regressor_col'


# Defaults Values
DEFAULT_REGRESSOR_SIGN = '='
DEFAULT_REGRESSOR_BETA = 0
DEFAULT_REGRESSOR_SIGMA = 1.0

# beta coef columns
COEFFICIENT_DF_COLS = namedtuple(
    'coefficients_df_cols',
    ['REGRESSOR', 'REGRESSOR_SIGN', 'COEFFICIENT']
)('regressor', 'regressor_sign', 'coefficient')
