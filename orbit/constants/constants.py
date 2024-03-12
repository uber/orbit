from collections import namedtuple
from enum import Enum


class PredictMethod(Enum):
    """
    The predict method for all of the stan template. Often used are mean and median.
    """

    MAP = "map"
    MEAN = "mean"
    MEDIAN = "median"
    FULL_SAMPLING = "full"


class PredictionKeys(Enum):
    """
    column names for the data frame of predicted result with decomposed components
    """

    TREND = "trend"
    SEASONALITY = "seasonality"
    REGRESSION = "regression"
    REGRESSOR = "regressor"
    PREDICTION = "prediction"


class EstimatorsKeys(Enum):
    """
    alias for all available estimator types when they are called under model wrapper functions
    """

    PyroSVI = "pyro-svi"
    StanMAP = "stan-map"
    StanMCMC = "stan-mcmc"
    CmdStanMAP = "cmdstan-map"
    CmdStanMCMC = "cmdstan-mcmc"


class TrainingMetaKeys(Enum):
    """
    training meta data dictionary processed under `Forecaster.fit()`
    """

    RESPONSE = "response"
    DATE_ARRAY = "date_array"
    NUM_OF_OBS = "num_of_obs"
    RESPONSE_SD = "response_sd"
    RESPONSE_MEAN = "response_mean"
    # START and END are in date-time format
    START = "training_start"
    END = "training_end"
    RESPONSE_COL = "response_col"
    DATE_COL = "date_col"


class PredictionMetaKeys(Enum):
    """
    prediction input meta data dictionary processed under `Forecaster.predict()`
    """

    DATE_ARRAY = "date_array"
    # FIXME: seems this is redudant? we can derive forecast range by input of data frame?
    # FIXME: or shall we cast this as zero when it is within training
    FUTURE_STEPS = "n_forecast_steps"
    # START and END are in date-time format
    START = "prediction_start"
    END = "prediction_end"
    START_INDEX = "start"
    END_INDEX = "end"
    PREDICTION_DF_LEN = "df_length"


class PlotLabels(Enum):
    """
    used in multiple prediction plots
    """

    # Also used in training_actual_response column name.
    TRAINING_ACTUAL_RESPONSE = "training_actual_response"
    PREDICTED_RESPONSE = "predicted_response"
    ACTUAL_RESPONSE = "actual_response"


class TimeSeriesSplitSchemeKeys(Enum):
    """hash table keys for the dictionary of back-test meta data"""

    MODEL = "model"
    TRAIN_START_DATE = "train_start_date"
    TRAIN_END_DATE = "train_end_date"
    TRAIN_IDX = "train_idx"
    TEST_IDX = "test_idx"
    # split scheme type
    SPLIT_TYPE_EXPANDING = "expanding"
    SPLIT_TYPE_ROLLING = "rolling"


class BacktestFitKeys(Enum):
    """column names of the dataframe used in the output from the backtest.BackTester.fit_predict() or any labels of
    the intermediate variables to generate such outcome dataframe
    """

    # labels for fitting process
    # note that the convention "_prediction" cannot be changed since it is also assumed in
    # all metric functions signature
    ACTUAL = "actual"
    PREDICTED = "prediction"
    DATE = "date"
    SPLIT_KEY = "split_key"
    TRAIN_FLAG = "training_data"
    TRAIN_ACTUAL = "train_actual"
    TRAIN_PREDICTED = "train_prediction"
    TEST_ACTUAL = "test_actual"
    TEST_PREDICTED = "test_prediction"
    # labels for scoring process
    METRIC_VALUES = "metric_values"
    METRIC_NAME = "metric_name"
    TRAIN_METRIC_FLAG = "is_training_metric"


class KTRTimePointPriorKeys(Enum):
    """hash table keys for the dictionary of back-test aggregation analysis result"""

    NAME = "name"
    PRIOR_START_TP_IDX = "prior_start_tp_idx"
    PRIOR_END_TP_IDX = "prior_end_tp_idx"
    PRIOR_MEAN = "prior_mean"
    PRIOR_SD = "prior_sd"
    PRIOR_REGRESSOR_COL = "prior_regressor_col"


# Defaults Values
DEFAULT_REGRESSOR_SIGN = "="
DEFAULT_REGRESSOR_BETA = 0
DEFAULT_REGRESSOR_SIGMA = 1.0

# beta coef columns
COEFFICIENT_DF_COLS = namedtuple(
    "coefficients_df_cols",
    ["REGRESSOR", "REGRESSOR_SIGN", "COEFFICIENT", "PROB_COEF_POS", "PROB_COEF_NEG"],
)("regressor", "regressor_sign", "coefficient", "Pr(coef >= 0)", "Pr(coef < 0)")


class CompiledStanModelPath:
    """
    the directory path for compliled stan models
    """

    PARENT = "orbit"
    CHILD = "stan_compiled"
