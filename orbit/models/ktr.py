from ..template.ktr import KTRModel
from ..forecaster import SVIForecaster
from ..exceptions import IllegalArgument
from ..estimators.pyro_estimator import PyroEstimatorVI


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
        # time-based coefficient priors
        coef_prior_list=None,
        # shared
        date_freq=None,
        flat_multiplier=True,
        # TODO: rename to residuals upper bound
        min_residuals_sd=1.0,
        ktrlite_optim_args=dict(),
        estimator='pyro-vi',
        **kwargs):
    """
    Args
    ----------
    level_knot_scale : float
        sigma for level; default to be .1
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
    span_coefficients : float between (0, 1)
        window width to decide the number of windows for the regression term
    rho_coefficients : float
        sigma in the Gaussian kernel for the regression term
    degree of freedom : int
        degree of freedom for error t-distribution
    coef_prior_list : list of dicts
        each dict in the list should have keys as
        'name', prior_start_tp_idx' (inclusive), 'prior_end_tp_idx' (not inclusive),
        'prior_mean', 'prior_sd', and 'prior_regressor_col'
    level_knot_dates : array like
        list of pre-specified dates for level knots
    level_knots : array like
        list of knot locations for level
        level_knot_dates and level_knots should be of the same length
    seasonal_knots_input : dict
         a dictionary for seasonality inputs with the following keys:
            '_seas_coef_knot_dates' : knot dates for seasonal regressors
            '_sea_coef_knot' : knot locations for sesonal regressors
            '_seasonality' : seasonality order
            '_seasonality_fs_order' : fourier series order for seasonality
    coefficients_knot_length : int
        the distance between every two knots for coefficients
    coefficients_knot_dates : array like
        a list of pre-specified knot dates for coefficients
    date_freq : str
        date frequency; if not supplied, pd.infer_freq will be used to imply the date frequency.
    min_residuals_sd : float
        a numeric value from 0 to 1 to indicate the upper bound of residual scale parameter; e.g.
        0.5 means residual scale will be sampled from [0, 0.5] in a scaled Beta(2, 2) dist.
    flat_multiplier : bool
        Default set as True. If False, we will adjust knot scale with a multiplier based on regressor volume
        around each knot; When True, set all multiplier as 1
    geometric_walk : bool
        Default set as False. If True we will sample positive regressor knot as geometric random walk
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
        additional arguments passed into orbit.estimators.stan_estimator or orbit.estimators.pyro_estimator
    """
    _supported_estimators = ['pyro-svi']

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
            min_residuals_sd=min_residuals_sd,
            ktrlite_optim_args=ktrlite_optim_args
    )
    if estimator == 'pyro-svi':
        ktr_forecaster = SVIForecaster(
            model=ktr,
            estimator_type=PyroEstimatorVI,
            **kwargs
        )
    else:
        raise IllegalArgument('Invalid estimator. Must be one of {}'.format(_supported_estimators))

    return ktr_forecaster
