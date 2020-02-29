from enum import Enum
import os

from orbit.utils.utils import get_parent_path


class PredictMethod(Enum):
    """
    The predict method for all of the stan models. Often used are mean and median.
    """
    MAP = 'map'
    MEAN = 'mean'
    MEDIAN = 'median'
    FULL_SAMPLING = 'full'


class SampleMethod(Enum):
    """
    The predict method for all of the stan models. Often used are mean and median.
    """
    VARIATIONAL_INFERENCE = 'vi'
    MARKOV_CHAIN_MONTE_CARLO = 'mcmc'


class PredictionColumnNames(Enum):
    """
    In the output of SLGTModel.transform() and SLGT.predict(), the column names if
    'return_decomposed_components' = True.
    """
    PREDICTED_RESPONSE = 'predicted'
    LEVEL = 'level'
    SEASONALITY = 'seasonality'
    REGRESSOR = 'regressor'
    ACTUAL_RESPONSE = 'actual'


class PlotLabels(Enum):
    # Also used in training_actual_response column name.
    TRAINING_ACTUAL_RESPONSE = 'training_actual_response'
    PREDICTED_RESPONSE = 'predicted_response'
    ACTUAL_RESPONSE = 'actual_response'


class StanModelKeys(Enum):
    """
    All of the keys in the trained stan model from uTS. For example, for LGT/SLGT,
    the model is the output of SLGT.fit() and input of SLGTModel.
    """
    STAN_INPUTS = "stan_inputs"
    MODELS = "models"
    REGRESSOR_COLUMNS = "regressor_columns"
    RESPONSE_COLUMN = "response_column"
    DATE_INFO = "date_info"


class DateInfo(Enum):
    """
    date_column: the data column name of the training/prediction data frame;
    starting_date: the date of first day of training data; format: yyyy-mm-dd
    date_interval: 'day', 'week', 'month'
    """
    DATE_COLUMN_NAME = 'date_column_name'
    DATE_COLUMN = 'date_column'
    START_DATE = 'start_date'
    END_DATE = 'end_date'
    DATE_INTERVAL = 'date_interval'


class BacktestMetaKeys(Enum):
    """ hash table keys for the dictionary of back-test meta data
    """
    MODEL = 'model'
    TRAIN_START_DATE = 'train_start_date'
    TRAIN_END_DATE = 'train_end_date'
    TRAIN_IDX = 'train_idx'
    TEST_IDX = 'test_idx'
    FORECAST_DATES = 'forecast_dates'


class BacktestFitColumnNames(Enum):
    """ column names for the data frame of back-test fitting result
    """
    TRAIN_START_DATE = 'train_start_date'
    TRAIN_END_DATE = 'train_end_date'
    FORECAST_DATES = 'forecast_dates'
    ACTUAL = 'actual'
    PRED = 'pred'
    PRED_HORIZON = 'pred_horizon'


class BacktestAnalyzeKeys(Enum):
    """ hash table keys for the dictionary of back-test aggregation analysis result
    """
    METRIC_NAME = 'metric_name'
    METRIC_PER_BTMOD = 'metric_per_btmod'
    METRIC_GEO = 'metric_geo'
    METRIC_PER_HORIZON = 'metric_per_horizon'


# Misc constants
THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
UTS_ROOT_DIR = get_parent_path(THIS_FILE_PATH)
SRC_DIR = os.path.join(UTS_ROOT_DIR, "src")  # pragma: no cover

# Defaults Values
DEFAULT_REGRESSOR_SIGN = '='
DEFAULT_REGRESSOR_BETA = 0
DEFAULT_REGRESSOR_SIGMA = 1.0