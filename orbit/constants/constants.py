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


class EstimatorsKeys(Enum):
    """
    alias for all available estimator types when they are called under model wrapper functions
    """
    PyroSVI = 'pyro-svi'
    StanMAP = 'stan-map'
    StanMCMC = 'stan-mcmc'


class TrainingMetaKeys(Enum):
    """
    training meta data dictionary processed under `Forecaster.fit()`
    """
    RESPONSE = 'response'
    DATE_ARRAY = 'date_array'
    NUM_OF_OBSERVATIONS = 'num_of_obs'
    RESPONSE_SD = 'response_sd'
    TRAINING_START = 'training_start'
    TRAINING_END = 'training_end'
    RESPONSE_COL = 'response_col'
    DATE_COL = 'date_col'


class PredictionMetaKeys(Enum):
    """
    prediction input meta data dictionary processed under `Forecaster.predict()`
    """
    NUM_OF_FORECAST_STEPS = 'n_forecast_steps'
    FORECAST_START = 'start'


class PlotLabels(Enum):
    """
    used in multiple prediction plots
    """
    # Also used in training_actual_response column name.
    TRAINING_ACTUAL_RESPONSE = 'training_actual_response'
    PREDICTED_RESPONSE = 'predicted_response'
    ACTUAL_RESPONSE = 'actual_response'


class TimeSeriesSplitSchemeKeys(Enum):
    """ hash table keys for the dictionary of back-test meta data
    """
    MODEL = 'model'
    TRAIN_START_DATE = 'train_start_date'
    TRAIN_END_DATE = 'train_end_date'
    TRAIN_IDX = 'train_idx'
    TEST_IDX = 'test_idx'


class BacktestFitKeys(Enum):
    """ column names for the data frame of back-test fitting result
    """
    ACTUAL = 'actual'
    PREDICTED = 'predicted'


class KTRTimePointPriorKeys(Enum):
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
