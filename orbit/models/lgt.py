# from inspect import signature
from ..template.lgt import LGTModel
from ..forecaster import MAPForecaster, FullBayesianForecaster, SVIForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP, StanEstimatorMCMC
from ..estimators.pyro_estimator import PyroEstimatorVI


def LGT(seasonality=None,
        seasonality_sm_input=None,
        level_sm_input=None,
        regressor_col=None,
        regressor_sign=None,
        regressor_beta_prior=None,
        regressor_sigma_prior=None,
        regression_penalty='fixed_ridge',
        lasso_scale=0.5,
        auto_ridge_scale=0.5,
        slope_sm_input=None,
        estimator='stan-mcmc',
        **kwargs):
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
    estimator : string

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

    **kwargs:
        additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
    """
    # ets_args_keys = [x for x in signature(ETSModel).parameters.keys() if x != 'kwargs']
    # ets_args = dict()
    # forecaster_args = dict()
    # for k, v in kwargs.items():
    #     if k in ets_args_keys:
    #         ets_args[k] = v
    #     else:
    #         forecaster_args[k] = v
    _supported_estimators = ['stan-map', 'stan-mcmc', 'pyro-svi']

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
    if estimator == 'stan-map':
        lgt_forecaster = MAPForecaster(
            model=lgt,
            estimator_type=StanEstimatorMAP,
            **kwargs
        )
    elif estimator == 'stan-mcmc':
        lgt_forecaster = FullBayesianForecaster(
            model=lgt,
            estimator_type=StanEstimatorMCMC,
            **kwargs
        )
    elif estimator == 'pyro-svi':
        lgt_forecaster = SVIForecaster(
            model=lgt,
            estimator_type=PyroEstimatorVI,
            **kwargs
        )
    else:
        raise IllegalArgument('Invalid estimator. Must be one of {}'.format(_supported_estimators))

    return lgt_forecaster
