# from inspect import signature
# from ..template.arma import ARMAModel
from orbit.template.arma import ARMAModel
from ..forecaster import MAPForecaster, FullBayesianForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP, StanEstimatorMCMC
from ..constants.constants import EstimatorsKeys


def ARMA(
    response_col,
    ar_lags=[],
    ma_lags=[],
    level_first=False,
    estimator="stan-mcmc",
    **kwargs,
):
    """
    Args
    ----------
    estimator : string; {'stan-mcmc', 'stan-map'}
        default to be 'stan-mcmc'.
    ar_lags : int; this is a list of the lags used in the ar model.
        Note that these do not need to be a full list; e.g., it could be [1,7, 365] and does not need to be 1 to 365!
    ma_lags : int; this is a list of the lags used in the ma model.
        Note that these do not need to be a full list; e.g., it could be [1,7, 365] and does not need to be 1 to 365!
    level_first: int; 1 or 0 to indicate if the mean should be removed from the signal before ARMA.
    **kwargs:
        additional arguments passed into orbit.estimators.stan_estimator
    """
    _supported_estimators = [
        EstimatorsKeys.StanMAP.value,
        EstimatorsKeys.StanMCMC.value,
    ]

    arma = ARMAModel(
        response_col=response_col,
        level_first=level_first,
        num_of_ar_lags=len(ar_lags),
        num_of_ma_lags=len(ma_lags),
        ar_lags=ar_lags,
        ma_lags=ma_lags,
    )
    if estimator == EstimatorsKeys.StanMAP.value:
        arma_forecaster = MAPForecaster(
            model=arma, estimator_type=StanEstimatorMAP, **kwargs
        )
    elif estimator == EstimatorsKeys.StanMCMC.value:
        arma_forecaster = FullBayesianForecaster(
            model=arma, estimator_type=StanEstimatorMCMC, **kwargs
        )
    else:
        raise IllegalArgument(
            "Invalid estimator. Must be one of {}".format(_supported_estimators)
        )

    return arma_forecaster
