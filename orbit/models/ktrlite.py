from ..template.ktrlite import KTRLiteModel
from ..forecaster import MAPForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP


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
        estimator='stan-map',
        **kwargs):
    """
    Args
    ----------
    seasonality : int, or list of int
        multiple seasonality
    seasonality_fs_order : int, or list of int
        fourier series order for seasonality
    level_knot_scale : float
        sigma for level; default to be .5
    seasonal_initial_knot_scale : float
        scale parameter for seasonal regressors initial coefficient knots; default to be 1
    seasonal_knot_scale : float
        scale parameter for seasonal regressors drift of coefficient knots; default to be 0.1.
    span_level : float between (0, 1)
        window width to decide the number of windows for the level (trend) term.
        e.g., span 0.1 will produce 10 windows.
    span_coefficients : float between (0, 1)
        window width to decide the number of windows for the regression term
    degree of freedom : int
        degree of freedom for error t-distribution
    level_knot_dates : array like
        list of pre-specified dates for the level knots
    level_knot_length : int
        the distance between every two knots for level
    coefficients_knot_length : int
        the distance between every two knots for coefficients
    knot_location : {'mid_point', 'end_point'}; default 'mid_point'
        knot locations. When level_knot_dates is specified, this is ignored for level knots.
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

    **kwargs:
        additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
    """
    _supported_estimators = ['stan-map']

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
    if estimator == 'stan-map':
        ktrlite_forecaster = MAPForecaster(
            model=ktrlite,
            estimator_type=StanEstimatorMAP,
            **kwargs
        )
    else:
        raise IllegalArgument('Invalid estimator. Must be one of {}'.format(_supported_estimators))

    return ktrlite_forecaster

