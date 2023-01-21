# from inspect import signature
from ..template.ets import ETSModel
from ..forecaster import MAPForecaster, FullBayesianForecaster
from ..exceptions import IllegalArgument
from ..constants.constants import EstimatorsKeys


def ETS(
    seasonality=None,
    seasonality_sm_input=None,
    level_sm_input=None,
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
    estimator : string; {'stan-mcmc', 'stan-map'}
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
        additional arguments passed into orbit.estimators.stan_estimator
    """
    _supported_estimators = [
        EstimatorsKeys.StanMAP.value,
        EstimatorsKeys.StanMCMC.value,
    ]

    ets = ETSModel(
        seasonality=seasonality,
        seasonality_sm_input=seasonality_sm_input,
        level_sm_input=level_sm_input,
    )
    if estimator == EstimatorsKeys.StanMAP.value:
        from ..estimators.stan_estimator import StanEstimatorMAP

        ets_forecaster = MAPForecaster(
            model=ets,
            estimator_type=StanEstimatorMAP,
            suppress_stan_log=suppress_stan_log,
            **kwargs,
        )
    elif estimator == EstimatorsKeys.StanMCMC.value:
        from ..estimators.stan_estimator import StanEstimatorMCMC

        ets_forecaster = FullBayesianForecaster(
            model=ets,
            estimator_type=StanEstimatorMCMC,
            suppress_stan_log=suppress_stan_log,
            **kwargs,
        )
    else:
        raise IllegalArgument(
            "Invalid estimator. Must be one of {}".format(_supported_estimators)
        )

    return ets_forecaster
