from ..template.ktrlite import KTRLiteModel
from ..forecaster import MAPForecaster
from ..exceptions import IllegalArgument
from ..estimators.stan_estimator import StanEstimatorMAP


def KTRLite(
        seasonality=None,
        seasonality_fs_order=None,
        level_knot_scale=0.5,
        seasonal_initial_knot_scale=1.0,
        seasonal_knot_scale=0.1,
        span_level=0.1,
        span_coefficients=0.3,
        degree_of_freedom=30,
        # knot customization
        level_knot_dates=None,
        level_knot_length=None,
        coefficients_knot_length=None,
        knot_location='mid_point',
        date_freq=None,
        estimator='stan-map',
        **kwargs):
    """
    Args
    ----------


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
        seasonality=seasonality,
        seasonality_fs_order=seasonality_fs_order,
        level_knot_scale=level_knot_scale,
        seasonal_initial_knot_scale=seasonal_initial_knot_scale,
        seasonal_knot_scale=seasonal_knot_scale,
        span_level=span_level,
        span_coefficients=span_coefficients,
        degree_of_freedom=degree_of_freedom,
        # knot customization
        level_knot_dates=level_knot_dates,
        level_knot_length=level_knot_length,
        coefficients_knot_length=coefficients_knot_length,
        knot_location=knot_location,
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

