from enum import Enum
import os

from orbit.utils.utils import get_parent_path


class StanInputMapper(Enum):
    """
    mapping from object input to stan file
    """
    # All of the following have default defined in DEFAULT_SLGT_FIT_ATTRIBUTES
    # ----------  Data Input ---------- #
    # observation related
    NUM_OF_OBSERVATIONS = 'NUM_OF_OBS'
    RESPONSE = 'RESPONSE'
    # ---------- Seasonality ---------- #
    SEASONALITY = 'SEASONALITY'
    SEASONALITY_MIN = 'SEA_MIN'
    SEASONALITY_MAX = 'SEA_MAX'
    SEASONALITY_SMOOTHING_MIN = 'SEA_SM_MIN'
    SEASONALITY_SMOOTHING_MAX = 'SEA_SM_MAX'
    # ---------- Trend ---------- #
    # LGT Trend
    GLOBAL_TREND_COEF_MIN = 'GT_COEF_MIN'
    GLOBAL_TREND_COEF_MAX = 'GT_COEF_MAX'
    GLOBAL_TREND_POW_MIN = 'GT_POW_MIN'
    GLOBAL_TREND_POW_MAX = 'GT_POW_MAX'
    LOCAL_TREND_COEF_MIN = 'LT_COEF_MIN'
    LOCAL_TREND_COEF_MAX = 'LT_COEF_MAX'
    LEVEL_SMOOTHING_MIN = 'LEV_SM_MIN'
    LEVEL_SMOOTHING_MAX = 'LEV_SM_MAX'
    SLOPE_SMOOTHING_MIN = 'SLP_SM_MIN'
    SLOPE_SMOOTHING_MAX = 'SLP_SM_MAX'
    # Damped Local Trend
    USE_DAMPED_TREND = 'USE_DAMPED_TREND'
    DAMPED_FACTOR_MIN = 'DAMPED_FACTOR_MIN'
    DAMPED_FACTOR_MAX = 'DAMPED_FACTOR_MAX'
    DAMPED_FACTOR_FIXED = 'DAMPED_FACTOR_FIXED'
    # ----------  Residuals  ---------- #
    MIN_NU = 'MIN_NU'
    MAX_NU = 'MAX_NU'
    CAUCHY_SD = 'CAUCHY_SD'
    # ----------  Regressions ---------- #
    FIX_REGRESSION_COEF_SD = 'FIX_REG_COEF_SD'
    REGRESSOR_SIGMA_SD = 'REG_SIGMA_SD'
    REGRESSION_COEF_MAX = 'BETA_MAX'
    NUM_OF_POSITIVE_REGRESSORS = 'NUM_OF_PR'
    POSITIVE_REGRESSOR_MATRIX = 'PR_MAT'
    POSITIVE_REGRESSOR_BETA_PRIOR = 'PR_BETA_PRIOR'
    POSITIVE_REGRESSOR_SIGMA_PRIOR = 'PR_SIGMA_PRIOR'
    NUM_OF_REGULAR_REGRESSORS = 'NUM_OF_RR'
    REGULAR_REGRESSOR_MATRIX = 'RR_MAT'
    REGULAR_REGRESSOR_BETA_PRIOR = 'RR_BETA_PRIOR'
    REGULAR_REGRESSOR_SIGMA_PRIOR = 'RR_SIGMA_PRIOR'


class LocalTrendStanSamplingParameters(Enum):
    LOCAL_TREND_LEVELS = 'l'
    LOCAL_TREND_SLOPES = 'b'
    RESIDUAL_SIGMA = 'obs_sigma'
    RESIDUAL_DEGREE_OF_FREEDOM = 'nu'
    LEVEL_SMOOTHING_FACTOR = 'lev_sm'
    SLOPE_SMOOTHING_FACTOR = 'slp_sm'


class GlobalTrendStanSamplingParameters(Enum):
    """
    The most basic stan output sampling parameters, local + global + trend model.
    """
    LOCAL_GLOBAL_TREND_SUMS = 'lgt_sum'
    GLOBAL_TREND_POWER = 'gt_pow'
    LOCAL_TREND_COEF = 'lt_coef'
    GLOBAL_TREND_COEF = 'gt_coef'


class DampedTrendDynamicStanSamplingParameters(Enum):
    DAMPED_FACTOR = 'damped_factor'


class DampedTrendStanSamplingParameters(Enum):
    LOCAL_TREND = 'lt_sum'
    GLOBAL_TREND = 'gt_sum'
    GLOBAL_TREND_SLOPE = 'gb'
    GLOBAL_TREND_LEVEL = 'gl'


class RegressionStanSamplingParameters(Enum):
    """
    The stan output sampling parameters related with regression component.
    """
    POSITIVE_REGRESSOR_BETA = 'pr_beta'
    REGULAR_REGRESSOR_BETA = 'rr_beta'


class SeasonalityStanSamplingParameters(Enum):
    """
    The stan output sampling parameters related with seasonality component.
    """
    SEASONALITY_LEVELS = 's'
    # we don't need initial seasonality
    # INITIAL_SEASONALITY = 'init_sea'
    SEASONALITY_SMOOTHING_FACTOR = 'sea_sm'


class AdstockStanSamplingParameters(Enum):
    """
    The stan output sampling parameters related with adstock
    """
    ADSTOCK_WEIGHTS = 'theta'


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