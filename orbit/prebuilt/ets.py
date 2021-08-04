from inspect import signature
from ..models.ets import ETSModel
from ..forecaster import MAPForecaster, FullBayesianForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP, StanEstimatorMCMC


def ETS(response_col='y', date_col='ds', estimator='stan-map', **kwargs):
    """
    Args
    ----------
    response_col : str
        Name of response variable column, default 'y'
    date_col : str
        Name of date variable column, default 'ds'
    seasonality : int
        Length of seasonality
    seasonality_sm_input : float
        float value between [0, 1], applicable only if `seasonality` > 1. A larger value puts
        more weight on the current seasonality.
        If None, the model will estimate this value.
    level_sm_input : float
        float value between [0.0001, 1]. A larger value puts more weight on the current level.
        If None, the model will estimate this value.
    estimator: string

    Other Parameters
    ----------------
    **kwargs: additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
    """
    ets_args_keys = [x for x in signature(ETSModel).parameters.keys() if x != 'kwargs']
    ets_args = dict()
    forecaster_args = dict()
    for k, v in kwargs.items():
        if k in ets_args_keys:
            ets_args[k] = v
        else:
            forecaster_args[k] = v
    ets = ETSModel(**ets_args)
    if estimator == 'stan-map':
        ets_forecaster = MAPForecaster(
            model=ets,
            response_col=response_col,
            date_col=date_col,
            estimator_type=StanEstimatorMAP,
            **forecaster_args
        )
    elif estimator == 'stan-mcmc':
        ets_forecaster = FullBayesianForecaster(
            model=ets,
            response_col=response_col,
            date_col=date_col,
            estimator_type=StanEstimatorMCMC,
            **forecaster_args
        )
    else:
        raise IllegalArgument('Invalid estimator.')

    return ets_forecaster


