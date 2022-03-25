from ..template.ktr import KTRModel
from ..forecaster import SVIForecaster
from ..exceptions import IllegalArgument
from ..estimators.pyro_estimator import PyroEstimatorSVI
from ..constants.constants import EstimatorsKeys


def KTR(
    level_knot_scale=0.1,
    level_segments=10,
    level_knot_distance=None,
    level_knot_dates=None,
    # seasonality
    seasonality=None,
    seasonality_fs_order=None,
    seasonality_segments=3,
    seasonal_initial_knot_scale=1.0,
    seasonal_knot_scale=0.1,
    # regression
    regressor_col=None,
    regressor_sign=None,
    regressor_init_knot_loc=None,
    regressor_init_knot_scale=None,
    regressor_knot_scale=None,
    regression_segments=5,
    regression_knot_distance=None,
    regression_knot_dates=None,
    # different from seasonality
    regression_rho=0.15,
    # shared
    degree_of_freedom=30,
    date_freq=None,
    # time-based coefficient priors
    coef_prior_list=None,
    flat_multiplier=True,
    # TODO: rename to residuals upper bound
    residuals_scale_upper=None,
    ktrlite_optim_args=dict(),
    estimator="pyro-svi",
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
    regressor_col : array-like strings
        regressor columns
    regressor_sign : list
        list of signs with '=' for regular regressor and '+' for positive regressor
    regressor_init_knot_loc : list
        list of regressor knot pooling mean priors, default to be 0's
    regressor_init_knot_scale : list
        list of regressor knot pooling sigma's to control the pooling strength towards the grand mean of regressors;
        default to be 1.
    regressor_knot_scale : list
        list of regressor knot sigma priors; default to be 0.1.
    regression_segments : int
        the number of segments partitioned by the knots of regression
    regression_knot_distance : int
        the distance between every two knots of regression
    regression_knot_dates : array-like
        list of pre-specified dates for regression knots
    regression_rho : float
        sigma in the Gaussian kernel for the regression term
    degree_of_freedom : int
        degree of freedom for error t-distribution
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    coef_prior_list : list of dicts
        each dict in the list should have keys as
        'name', prior_start_tp_idx' (inclusive), 'prior_end_tp_idx' (not inclusive),
        'prior_mean', 'prior_sd', and 'prior_regressor_col'
    residuals_scale_upper : float
    flat_multiplier : bool
        Default set as True. If False, we will adjust knot scale with a multiplier based on regressor volume
        around each knot; When True, set all multiplier as 1
    ktrlite_optim_args : dict
        the optimizing config for the ktrlite model (to fit level/seasonality). Default to be dict().
    estimator : string; {'pyro-svi'}

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
        additional arguments passed into orbit.estimators.pyro_estimator
    """
    _supported_estimators = [EstimatorsKeys.PyroSVI.value]

    ktr = KTRModel(
        level_knot_scale=level_knot_scale,
        level_segments=level_segments,
        level_knot_distance=level_knot_distance,
        level_knot_dates=level_knot_dates,
        seasonality=seasonality,
        seasonality_fs_order=seasonality_fs_order,
        seasonality_segments=seasonality_segments,
        seasonal_initial_knot_scale=seasonal_initial_knot_scale,
        seasonal_knot_scale=seasonal_knot_scale,
        regressor_col=regressor_col,
        regressor_sign=regressor_sign,
        regressor_init_knot_loc=regressor_init_knot_loc,
        regressor_init_knot_scale=regressor_init_knot_scale,
        regressor_knot_scale=regressor_knot_scale,
        regression_segments=regression_segments,
        regression_knot_distance=regression_knot_distance,
        regression_knot_dates=regression_knot_dates,
        regression_rho=regression_rho,
        degree_of_freedom=degree_of_freedom,
        coef_prior_list=coef_prior_list,
        date_freq=date_freq,
        flat_multiplier=flat_multiplier,
        residuals_scale_upper=residuals_scale_upper,
        ktrlite_optim_args=ktrlite_optim_args,
    )
    if estimator == EstimatorsKeys.PyroSVI.value:
        ktr_forecaster = SVIForecaster(
            model=ktr, estimator_type=PyroEstimatorSVI, **kwargs
        )
    else:
        raise IllegalArgument(
            "Invalid estimator. Must be one of {}".format(_supported_estimators)
        )

    return ktr_forecaster
