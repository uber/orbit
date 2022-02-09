# from inspect import signature
# from ..template.arma import ARMAModel
from orbit.template.arma import ARMAModel
from ..forecaster import MAPForecaster, FullBayesianForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP, StanEstimatorMCMC
from ..constants.constants import EstimatorsKeys


def ARMA(response_col,
        ar_lags = [],
        ma_lags = [],
        regressor_col=None,
        lm_first=False,
        estimator='stan-mcmc',   
        **kwargs):
    """
    Args
    ----------
    regressor_col : list
        Names of regressor columns, if any
    lm_first : boolean
        This indicates if the ARMA model is on the residuals of the lm model or estimated concurrently 
    estimator : string; {'stan-mcmc', 'stan-map'}
        default to be 'stan-mcmc'.

    **kwargs:
        additional arguments passed into orbit.estimators.stan_estimator
    """
    regressor_col
    if regressor_col is not None:
        raise IllegalArgument('Orbit ARMA no longer supports linear prediction! Please remove the regressor_col argument.')
    
    _supported_estimators = [EstimatorsKeys.StanMAP.value, EstimatorsKeys.StanMCMC.value]

    arma = ARMAModel(
        regressor_col=regressor_col,
        lm_first = lm_first,
        
        num_of_ar_lags = len(ar_lags),
        num_of_ma_lags = len(ma_lags),
        ar_lags = ar_lags,
        ma_lags = ma_lags,
        response_col= response_col,
        
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
