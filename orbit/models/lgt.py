# from inspect import signature
from ..template.lgt import LGTModel
from ..forecaster import MAPForecaster, FullBayesianForecaster, SVIForecaster
from ..exceptions import IllegalArgument
from ..constants.constants import EstimatorsKeys


def LGT(
    seasonality=None,
    seasonality_sm_input=None,
    level_sm_input=None,
    regressor_col=None,
    regressor_sign=None,
    regressor_beta_prior=None,
    regressor_sigma_prior=None,
    regression_penalty="fixed_ridge",
    lasso_scale=0.5,
    auto_ridge_scale=0.5,
    slope_sm_input=None,
    estimator="stan-mcmc",
    suppress_stan_log=True,
    **kwargs,
):
    """
    Args
    ----------
    seasonality : int
        Length of seasonality
    seasonality_sm_input : float
        float value between [0, 1], applicable only if `seasonality` > 1. A larger value puts
        more weight on the current seasonality.
        If None, the model will estimate this value.
    level_sm_input : float
        float value between [0.0001, 1]. A larger value puts more weight on the current level.
        If None, the model will estimate this value.
    regressor_col : list
        Names of regressor columns, if any
    regressor_sign :  list
        list with values { '+', '-', '=' } such that
        '+' indicates regressor coefficient estimates are constrained to [0, inf).
        '-' indicates regressor coefficient estimates are constrained to (-inf, 0].
        '=' indicates regressor coefficient estimates can be any value between (-inf, inf).
        The length of `regressor_sign` must be the same length as `regressor_col`. If None,
        all elements of list will be set to '='.
    regressor_beta_prior : list
        list of prior float values for regressor coefficient betas. The length of `regressor_beta_prior`
        must be the same length as `regressor_col`. If None, use non-informative priors.
    regressor_sigma_prior : list
        list of prior float values for regressor coefficient sigmas. The length of `regressor_sigma_prior`
        must be the same length as `regressor_col`. If None, use non-informative priors.
    regression_penalty : { 'fixed_ridge', 'lasso', 'auto_ridge' }
        regression penalty method
    lasso_scale : float
        float value between [0, 1], applicable only if `regression_penalty` == 'lasso'
    auto_ridge_scale : float
        float value between [0, 1], applicable only if `regression_penalty` == 'auto_ridge'
    slope_sm_input : float
        float value between [0, 1]. A larger value puts more weight on the current slope.
        If None, the model will estimate this value.
    estimator : string; {'stan-mcmc', 'stan-map', 'pyro-svi'}
        default to be 'stan-mcmc'.

    Other Parameters
    ----------------
    response_col : str
        Name of response variable column, default 'y'
    date_col : str
        Name of date variable column, default 'ds'
    n_bootstrap_draws : int
        Number of samples to bootstrap in order to generate the prediction interval. For full Bayesian and
        variational inference forecasters, samples are drawn directly from original posteriors. For point-estimated
        posteriors, it will be used to sample noise parameters.  When -1 or None supplied, full Bayesian and
        variational inference forecasters will assume number of draws equal the size of original samples while
        point-estimated posteriors will mute the draw and output prediction without interval.
    prediction_percentiles : list
        List of integers of prediction percentiles that should be returned on prediction. To avoid reporting any
        confident intervals, pass an empty list
    suppress_stan_log : bool
        If False, turn off cmdstanpy logger. Default as False.

    **kwargs:
        additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
    """
    _supported_estimators = [
        EstimatorsKeys.StanMAP.value,
        EstimatorsKeys.StanMCMC.value,
        EstimatorsKeys.PyroSVI.value,
    ]

    lgt = LGTModel(
        seasonality=seasonality,
        seasonality_sm_input=seasonality_sm_input,
        level_sm_input=level_sm_input,
        regressor_col=regressor_col,
        regressor_sign=regressor_sign,
        regressor_beta_prior=regressor_beta_prior,
        regressor_sigma_prior=regressor_sigma_prior,
        regression_penalty=regression_penalty,
        lasso_scale=lasso_scale,
        auto_ridge_scale=auto_ridge_scale,
        slope_sm_input=slope_sm_input,
    )
    if estimator == EstimatorsKeys.StanMAP.value:
        from ..estimators.stan_estimator import StanEstimatorMAP

        lgt_forecaster = MAPForecaster(
            model=lgt,
            estimator_type=StanEstimatorMAP,
            suppress_stan_log=suppress_stan_log,
            **kwargs,
        )
    elif estimator == EstimatorsKeys.StanMCMC.value:
        from ..estimators.stan_estimator import StanEstimatorMCMC

        lgt_forecaster = FullBayesianForecaster(
            model=lgt,
            estimator_type=StanEstimatorMCMC,
            suppress_stan_log=suppress_stan_log,
            **kwargs,
        )
    elif estimator == EstimatorsKeys.PyroSVI.value:
        from ..estimators.pyro_estimator import PyroEstimatorSVI

        lgt_forecaster = SVIForecaster(
            model=lgt, estimator_type=PyroEstimatorSVI, **kwargs
        )
    else:
        raise IllegalArgument(
            "Invalid estimator. Must be one of {}".format(_supported_estimators)
        )

    return lgt_forecaster
