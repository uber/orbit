# from inspect import signature
# from ..template.arma import ARMAModel
from orbit.template.arma import ARMAModel
from ..forecaster import MAPForecaster, FullBayesianForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP, StanEstimatorMCMC
from ..constants.constants import EstimatorsKeys


def ARMA(response_col,
         ar_lags=[],
         ma_lags=[],
         level_first=False,
         estimator='stan-mcmc',
         **kwargs):
    """
    Args
    ----------
    estimator : string; {'stan-mcmc', 'stan-map'}
        default to be 'stan-mcmc'.

    **kwargs:
        additional arguments passed into orbit.estimators.stan_estimator
    """
    _supported_estimators = [EstimatorsKeys.StanMAP.value, EstimatorsKeys.StanMCMC.value]

    arma = ARMAModel(
        response_col=response_col,
        lm_first=level_first,
        num_of_ar_lags=len(ar_lags),
        num_of_ma_lags=len(ma_lags),
        ar_lags=ar_lags,
        ma_lags=ma_lags,

    )
    if estimator == EstimatorsKeys.StanMAP.value:
        arma_forecaster = MAPForecaster(
            model=arma,
            estimator_type=StanEstimatorMAP,
            **kwargs
        )
    elif estimator == EstimatorsKeys.StanMCMC.value:
        arma_forecaster = FullBayesianForecaster(
            model=arma,
            estimator_type=StanEstimatorMCMC,
            **kwargs
        )
    else:
        raise IllegalArgument('Invalid estimator. Must be one of {}'.format(_supported_estimators))

    return arma_forecaster
