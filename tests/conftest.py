import pytest
import pandas as pd
import numpy as np
import pkg_resources

from .utils.utils import make_synthetic_series


@pytest.fixture
def iclaims_training_data():
    test_file = pkg_resources.resource_filename(
        'tests',
        'resources/iclaims.example.csv'
    )
    df = pd.read_csv(
        test_file,
        parse_dates=['week']
    )
    return df


@pytest.fixture
def m3_monthly_data():
    test_file = pkg_resources.resource_filename(
        'tests',
        'resources/m3-monthly.csv'
    )
    df = pd.read_csv(
        test_file,
        parse_dates=['date']
    )
    return df


@pytest.fixture
def synthetic_data():
    df, coef = make_synthetic_series(seed=127)

    train_df = df[df['week'] <= '2019-01-01']
    test_df = df[df['week'] > '2019-01-01']

    return train_df, test_df, coef


@pytest.fixture
def valid_sample_predict_method_combo():
    valid_permutations = [
        ("map", "map"),
        ("vi", "mean"), ("vi", "median"), ("vi", "full"),
        ("mcmc", "mean"), ("mcmc", "median"), ("mcmc", "full")
    ]
    return valid_permutations


@pytest.fixture
def valid_pyro_sample_predict_method_combo():
    valid_permutations = [
        ("map", "map"),
        ("vi", "mean"), ("vi", "median"), ("vi", "full")
    ]
    return valid_permutations


@pytest.fixture
def stan_estimator_lgt_model_input():
    stan_model_name = 'lgt'
    model_param_names = [
        'l', 'b', 'lev_sm', 'slp_sm', 'obs_sigma', 'nu',
        'lgt_sum', 'gt_pow', 'lt_coef', 'gt_coef', 's', 'sea_sm',
    ]
    data_input = {
        'WITH_MCMC': 1,
        'NUM_OF_OBS': 157,
        'RESPONSE': np.array([8.4176585, 8.64034557, 8.62200847, 8.4645145, 8.53425815,
                              8.72077311, 8.47047121, 8.66371487, 8.48439717, 8.60694622,
                              8.67657416, 8.77962136, 8.619585, 8.49614735, 8.73995347,
                              8.86694259, 8.65006437, 8.47589629, 8.52308679, 8.37466204,
                              8.53929883, 8.33839384, 8.11751237, 8.36545509, 8.29504817,
                              7.92784775, 8.26273662, 8.23594167, 7.96156787, 8.05358342,
                              8.34646046, 8.33540452, 8.00304161, 8.47621054, 8.28891251,
                              8.35915504, 7.93359056, 8.39664542, 8.42117692, 7.81848792,
                              7.93616947, 7.89520109, 7.83619989, 7.95084942, 7.27839095,
                              7.77222415, 7.82744773, 7.95244009, 7.9279573, 8.06028933,
                              8.30860513, 8.12826528, 7.90220128, 8.2897354, 8.3337894,
                              8.06080339, 8.45450729, 8.33476913, 8.41496814, 8.3802148,
                              8.34737954, 8.43220872, 8.46176215, 8.34754922, 8.3799409,
                              8.41140785, 8.58459938, 8.30785843, 8.38260875, 8.41704154,
                              8.4349066, 8.29964521, 8.21604235, 7.86840269, 8.14730503,
                              8.15060573, 8.22519524, 7.96496838, 7.83459426, 8.16690377,
                              7.92797669, 7.86785444, 8.14090668, 8.22738765, 8.06789266,
                              8.0944723, 8.07463537, 8.09946072, 8.02672951, 7.89728251,
                              7.70927749, 7.73805405, 7.36821146, 7.72032935, 7.58771258,
                              7.29146682, 7.80363816, 7.51267038, 7.99874461, 7.55528498,
                              7.7125198, 7.71091572, 7.8151259, 7.92529138, 7.81512862,
                              7.8711069, 7.8801177, 8.21061382, 8.13363296, 8.10531478,
                              7.98185624, 8.11712744, 8.06752991, 8.14161123, 8.09942204,
                              8.12278759, 8.30482629, 8.16593318, 7.86516153, 8.14815287,
                              8.10534245, 8.12602089, 7.94896453, 7.97067553, 7.67404384,
                              7.86055081, 7.6549576, 7.94651151, 7.5037856, 7.70004575,
                              7.69887184, 7.84241561, 7.84181436, 7.4587219, 7.55741971,
                              7.59392022, 7.74619782, 7.82054086, 7.70546586, 7.79190798,
                              7.40024447, 7.25319198, 7.6752235, 7.30917242, 7.33591809,
                              7.62541756, 7.35301334, 7.46718606, 6.97838846, 7.16976506,
                              7.32056103, 7.32163764, 7.40627366, 7.34321953, 7.26235866,
                              7.62360629, 7.61622702]),
        'SEASONALITY': 52,
        'SEA_SM_INPUT': -1,
        'LEV_SM_INPUT': -1,
        'SLP_SM_INPUT': -1,
        'MIN_NU': 5.0,
        'MAX_NU': 40.0,
        'CAUCHY_SD': 0.2955647529713108,
        'NUM_OF_PR': 0,
        'PR_MAT': np.empty((157, 0), dtype=np.float64),
        'PR_BETA_PRIOR': [],
        'PR_SIGMA_PRIOR': [],
        'NUM_OF_NR': 0,
        'NR_MAT': np.empty((157, 0), dtype=np.float64),
        'NR_BETA_PRIOR': [],
        'NR_SIGMA_PRIOR': [],
        'NUM_OF_RR': 0,
        'RR_MAT': np.empty((157, 0), dtype=np.float64),
        'RR_BETA_PRIOR': [],
        'RR_SIGMA_PRIOR': [],
        'REG_PENALTY_TYPE': 0,
        'AUTO_RIDGE_SCALE': 0.5,
        'LASSO_SCALE': 0.
    }
    return stan_model_name, model_param_names, data_input
