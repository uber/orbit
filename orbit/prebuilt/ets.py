from ..models.ets import ETSModel
from ..forecaster import MAPForecaster, FullBayesianForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP, StanEstimatorMCMC


def ETS(response_col, date_col, seasonality, seasonality_sm_input, level_sm_input, estimator='stan-map', **kwargs):
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
    ets = ETSModel(seasonality=seasonality, seasonality_sm_input=seasonality_sm_input, level_sm_input=level_sm_input)
    if estimator == 'stan-map':
        ets_forecaster = MAPForecaster(
            model=ets,
            response_col=response_col,
            date_col=date_col,
            estimator_type=StanEstimatorMAP,
            **kwargs
        )
    elif estimator == 'stan-mcmc':
        ets_forecaster = FullBayesianForecaster(
            model=ets,
            response_col=response_col,
            date_col=date_col,
            estimator_type=StanEstimatorMCMC,
            **kwargs
        )
    else:
        raise IllegalArgument('Invalid estimator.')

    return ets_forecaster


