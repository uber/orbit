from uTS.utils.constants import PredictMethod
from uTS.utils.utils_base import merge_two_dictionaries

"""
The key:value pairs of all possible attributes for uTS, shared by
LGT/SLGT, MultiSLGT, LGTAdstock.
    num_warmup_iter: int
        number of iterations for warmup;
    num_sample_iter: int
        number of sampling iteration, the total iteration = num_warmup_iter + num_sample_iter
         if there is only one chain.
    chains: int
        number of chains
    save_mcmc_sample: boolean
        logical variable whether save mcmc samples or not
    verbose: boolean
        logical variable whether to show certain messages or not.
"""
DEFAULT_MCMC_ATTRIBUTES = {
    'num_warmup_iter': 900,
    'num_sample_iter': 100,
    'chains': 4,
    'cores': 4,
    'seed': 6789,
    'save_mcmc_sample': False,
    'verbose': False
}

"""
The key:value pairs of all possible attributes for LGT.
    cauchy_sd: float
        a numeric value is usually set to max(responses) / 100
        if it is not pre-defined. Default is None.
    min_nu: float
    max_nu: float
    global_trend_coef_min: float
        minimum bound of global trend coefficient
    global_trend_coef_max: float
        maximum bound of global trend coefficient
    global_trend_pow_min: float
        minimum bound of global trend power
    global_trend_pow_max: float
        maximum bound of global trend power
    local_trend_coef_min: float
        minimum bound of local trend coefficient
    local_trend_coef_max: float
        maximum  bound of local trend coefficient
    level_smoothing_min: float
        minimum bound of level smoothing factor
    level_smoothing_max: float
        maximum bound of level smoothing factor
    slope_smoothing_min: float
        minimum bound of slope smoothing factor
    slope_smoothing_max: float
        maximum bound of slope smoothing factor
    regressor_columns: list
        a list of strings to represent columns used in regression.
    regressor_signs: list
        a list of either "+" or "=" to represent whether
        each column is positive regressor or regular regressor.
        It should have same length as regressor_columns.
        If it is not defined, then use ["="] * len(regressor_columns).
    regressor_beta_priors: list
        a list of numerical values to represent the beta priors usedfor each column.
        It should have same length as regressor_columns.
        If it is not defined, then use [0] * len(regressor_columns).
    regressor_sigma_priors: list
        a list of numerical values to represent the sigma priors used for each column.
        It should have same length as regressor_columns.
        If it is not defined, then use [1] * len(regressor_columns).
    use_regressor_cauchy: int, 0 or 1
        a binary value to represent whether to use cauchy regressor.
    regressor_cauchy_sd: float
        standard deviation of cauchy regressor
    beta_max: float
    use_damped_trend: 0 or 1
        a binary value to represent whether to use damped local trend
        instead of global trend
    damped_factor_min:
        a numerical value 0 to 1; must be less than damped_factor_max;
        this value indicates the minimum bound of damped factor to be estimated
    damped_factor_max:
        a numerical value 0 to 1; must be greater than damped_factor_min;
        this value indicates the minimum bound of damped factor to be estimated
    damped_factor_fixed:
        a numerical value 0 to 1; if this is > 0 and < 1 and use_damped_trend = 1,
        then we will use this as the deterministic
        value of damped factor instead of mcmc samples.
"""
DEFAULT_LGT_FIT_ATTRIBUTES = {
    # observation related
    'cauchy_sd': None,
    'min_nu': 5,
    'max_nu': 40,
    # hyper parameters min-max
    'global_trend_coef_min': -0.5,
    'global_trend_coef_max': 0.5,
    'global_trend_pow_min': 0.0,
    'global_trend_pow_max': 1,
    'local_trend_coef_min': 0,
    'local_trend_coef_max': 1,
    'level_smoothing_min': 0,
    'level_smoothing_max': 1,
    'slope_smoothing_min': 0,
    'slope_smoothing_max': 1,
    # regressions
    'regressor_columns': [],
    'regressor_signs': [],
    'regressor_beta_priors': [],
    'regressor_sigma_priors': [],
    'use_regressor_cauchy': 0,
    'regressor_cauchy_sd': 1.0,
    'beta_max': 1.1,
    # damped trend
    'use_damped_trend': 0,
    'damped_factor_min': 0.8,
    'damped_factor_max': 0.999,
    'damped_factor_fixed': 0.9
}

"""
The key:value pairs of all possible attributes for SLGT.
In addition to DEFAULT_LGT_FIT_ATTRIBUTES above, we add:
    seasonality: int
        a numeric value to indicate the seasonality. default assumes no seasonality i.e. (=1).
        For example, if our data is per week, and the period is a year, then seasonality = 52.
    seasonality_min: float
    seasonality_max: float
    seasonality_smoothing_min: float
        minimum bound of seasonality smoothing factor
    seasonality_smoothing_max: float
        maximum bound of seasonality smoothing factor
"""
DEFAULT_SLGT_FIT_ATTRIBUTES = merge_two_dictionaries(DEFAULT_LGT_FIT_ATTRIBUTES, {
    # seasonality related
    'seasonality': 1,
    'seasonality_min': -0.5,
    'seasonality_max': 0.5,
    'seasonality_smoothing_min': 0,
    'seasonality_smoothing_max': 1,
})

"""
The key:value pairs of all possible attributes for LGTAdstock.
In addition to DEFAULT_LGT_FIT_ATTRIBUTES above, we add:
"""
DEFAULT_LGT_ADSTOCK_FIT_ATTRIBUTES = merge_two_dictionaries(DEFAULT_LGT_FIT_ATTRIBUTES, {
    'adstock_transformation_scalars': [],  # scalar s such that x' = log(1/s + 1),
                                           # length = number_of_adstocks
    'length_of_adstock_weights': [],  # length of each adstock weight vectors,
                                      # length = number_of_adstocks
    'min_spike': 0.1,
    'max_spike': 0.4,
    'min_amplitude': 1,
    'max_amplitude': 3
})

"""
The key:value pairs of all possible attributes for SLGT and LGTAdstock predict().
    predict_method: str
        One of four predict methods to use for the prediction: mean, median, mode, mcmc.
    shift: integer
        the time offset between the predicting data and training data. For example,
        if training data starts from 2010-01-01, and predicting data starts
        from 2010-01-29. If the interval is 'day', then shift = 28;
        if the interval is 'week', shift = 4.
    return_decomposed_components: boolean
        Whether to return level, regression and seasonality components.
    simulate_residuals: boolean
        Whether to simulate residuals, only available in 'mcmc' predict method.
    n_boostrap_draws: int
        the number of bootstrap samples to draw from the posterior distribution
        If default, no bootstrap samples are drawn, and instead we use the
        previously sampled posterior during model fit
    prediction_interval_quantiles: list
        upper and lower quantile to use as prediction interval bounds
"""
DEFAULT_LGT_PREDICT_ATTRIBUTES = {
    'predict_method': PredictMethod.MEAN.value,
    'shift': 0,
    'return_decomposed_components': False,
    'n_bootstrap_draws': -1,
    'prediction_interval_quantiles': [.05, 0.95]
}
