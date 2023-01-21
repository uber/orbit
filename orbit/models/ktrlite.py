from ..template.ktrlite import KTRLiteModel
from ..forecaster import MAPForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP
from ..constants.constants import EstimatorsKeys


def KTRLite(
    # level
    level_knot_scale=0.1,
    level_segments=10,
    level_knot_distance=None,
    level_knot_dates=None,
    # seasonality
    seasonality=None,
    seasonality_fs_order=None,
    seasonality_segments=2,
    seasonal_initial_knot_scale=1.0,
    seasonal_knot_scale=0.1,
    degree_of_freedom=30,
    date_freq=None,
    estimator="stan-map",
    suppress_stan_log=True,
    **kwargs,
):
    """
    Parameters
    ----------
    level_knot_scale : float
        sigma for level; default to be .1
    level_segments : int
        the number of segments partitioned by the knots of level (trend)
    level_knot_distance : int
        the distance between every two knots of level (trend)
    level_knot_dates : array like
        list of pre-specified dates for the level knots
    seasonality : int, or list of int
        multiple seasonality
    seasonality_fs_order : int, or list of int
        fourier series order for seasonality
    seasonality_segments : int
        the number of segments partitioned by the knots of seasonality
    seasonal_initial_knot_scale : float
        scale parameter for seasonal regressors initial coefficient knots; default to be 1
    seasonal_knot_scale : float
        scale parameter for seasonal regressors drift of coefficient knots; default to be 0.1.
    degree_of_freedom : int
        degree of freedom for error t-distribution
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    estimator : string; {'stan-map'}

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
    _supported_estimators = [EstimatorsKeys.StanMAP.value]

    ktrlite = KTRLiteModel(
        level_knot_scale=level_knot_scale,
        level_segments=level_segments,
        level_knot_distance=level_knot_distance,
        level_knot_dates=level_knot_dates,
        # seasonality
        seasonality=seasonality,
        seasonality_fs_order=seasonality_fs_order,
        seasonality_segments=seasonality_segments,
        seasonal_initial_knot_scale=seasonal_initial_knot_scale,
        seasonal_knot_scale=seasonal_knot_scale,
        degree_of_freedom=degree_of_freedom,
        date_freq=date_freq,
    )
    if estimator == EstimatorsKeys.StanMAP.value:
        ktrlite_forecaster = MAPForecaster(
            model=ktrlite,
            estimator_type=StanEstimatorMAP,
            suppress_stan_log=suppress_stan_log,
            **kwargs,
        )
    else:
        raise IllegalArgument(
            "Invalid estimator. Must be one of {}".format(_supported_estimators)
        )

    return ktrlite_forecaster
