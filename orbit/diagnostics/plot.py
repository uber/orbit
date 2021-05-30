# the following lines are added to fix unit test error
# or else the following line will give the following error
# TclError: no display name and no $DISPLAY environment variable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from copy import deepcopy
import arviz as az

from orbit.constants.constants import PredictionKeys
from orbit.utils.general import is_empty_dataframe, is_ordered_datetime
from orbit.constants.palette import QualitativePalette

az.style.use("arviz-darkgrid")


def plot_predicted_data(training_actual_df, predicted_df, date_col, actual_col,
                        pred_col=PredictionKeys.PREDICTION.value, prediction_percentiles=None,
                        title="", test_actual_df=None, is_visible=True,
                        figsize=None, path=None, fontsize=None,
                        line_plot=False, markersize=70, lw=2, linestyle='-'):
    """
    plot training actual response together with predicted data; if actual response of predicted
    data is there, plot it too.
    Parameters
    ----------
    training_actual_df : pd.DataFrame
        training actual response data frame. two columns required: actual_col and date_col
    predicted_df : pd.DataFrame
        predicted data response data frame. two columns required: actual_col and pred_col. If
        user provide prediction_percentiles, it needs to include them as well in such
        `prediction_{x}` where x is the correspondent percentiles
    prediction_percentiles : list
        list of two elements indicates the lower and upper percentiles
    date_col : str
        the date column name
    actual_col : str
    pred_col : str
    title : str
        title of the plot
    test_actual_df : pd.DataFrame
       test actual response dataframe. two columns required: actual_col and date_col
    is_visible : boolean
        whether we want to show the plot. If called from unittest, is_visible might = False.
    figsize : tuple
        figsize pass through to `matplotlib.pyplot.figure()`
    path : str
        path to save the figure
    fontsize : int; optional
        fontsize of the title
    line_plot : bool; default False
        if True, make line plot for observations; otherwise, make scatter plot for observations
    markersize : int; optional
        point marker size
    lw : int; optional
        out-of-sample prediction line width
    linestyle : str
        linestyle of prediction plot
    Returns
    -------
        matplotlib axes object
    """

    if is_empty_dataframe(training_actual_df) or is_empty_dataframe(predicted_df):
        raise ValueError("No prediction data or training response to plot.")

    if not is_ordered_datetime(predicted_df[date_col]):
        raise ValueError("Prediction df dates is not ordered.")

    plot_confid = False
    if prediction_percentiles is None:
        _pred_percentiles = [5, 95]
    else:
        _pred_percentiles = prediction_percentiles

    if len(_pred_percentiles) != 2:
        raise ValueError("prediction_percentiles has to be None or a list with length=2.")

    confid_cols = ['prediction_{}'.format(_pred_percentiles[0]),
                   'prediction_{}'.format(_pred_percentiles[1])]

    if set(confid_cols).issubset(predicted_df.columns):
        plot_confid = True

    if not figsize:
        figsize = (16, 8)

    if not fontsize:
        fontsize = 16

    _training_actual_df = training_actual_df.copy()
    _predicted_df = predicted_df.copy()
    _training_actual_df[date_col] = pd.to_datetime(_training_actual_df[date_col])
    _predicted_df[date_col] = pd.to_datetime(_predicted_df[date_col])

    fig, ax = plt.subplots(facecolor='w', figsize=figsize)

    if line_plot:
        ax.plot(_training_actual_df[date_col].values,
                _training_actual_df[actual_col].values,
                marker=None, color='black', lw=lw, label='train response', linestyle=linestyle)
    else:
        ax.scatter(_training_actual_df[date_col].values,
                   _training_actual_df[actual_col].values,
                   marker='.', color='black', alpha=0.8, s=markersize,
                   label='train response')

    ax.plot(_predicted_df[date_col].values,
            _predicted_df[pred_col].values,
            marker=None, color='#12939A', lw=lw, label=PredictionKeys.PREDICTION.value, linestyle=linestyle)

    # vertical line separate training and prediction
    if _training_actual_df[date_col].values[-1] < _predicted_df[date_col].values[-1]:
        ax.axvline(x=_training_actual_df[date_col].values[-1], color='#1f77b4', linestyle='--')

    if test_actual_df is not None:
        test_actual_df = test_actual_df.copy()
        test_actual_df[date_col] = pd.to_datetime(test_actual_df[date_col])
        if line_plot:
            ax.plot(test_actual_df[date_col].values,
                    test_actual_df[actual_col].values,
                    marker=None, color='#FF8C00', lw=lw, label='train response', linestyle=linestyle)
        else:
            ax.scatter(test_actual_df[date_col].values,
                       test_actual_df[actual_col].values,
                       marker='.', color='#FF8C00', alpha=0.8, s=markersize,
                       label='test response')

    # prediction intervals
    if plot_confid:
        ax.fill_between(_predicted_df[date_col].values,
                        _predicted_df[confid_cols[0]],
                        _predicted_df[confid_cols[1]],
                        facecolor='#42999E', alpha=0.5)

    ax.set_title(title, fontsize=fontsize)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.5)
    ax.legend()
    if path:
        fig.savefig(path)
    if is_visible:
        plt.show()
    else:
        plt.close()

    return ax


def plot_predicted_components(predicted_df, date_col, prediction_percentiles=None, plot_components=None,
                              title="", figsize=None, path=None, fontsize=None, is_visible=True):
    """ Plot predicted componenets with the data frame of decomposed prediction where components
    has been pre-defined as `trend`, `seasonality` and `regression`.
    Parameters
    ----------
    predicted_df : pd.DataFrame
        predicted data response data frame. two columns required: actual_col and pred_col. If
        user provide pred_percentiles_col, it needs to include them as well.
    date_col : str
        the date column name
    prediction_percentiles : list
        a list should consist exact two elements which will be used to plot as lower and upper bound of
        confidence interval
    plot_components : list
        a list of strings to show the label of components to be plotted; by default, it uses values in
        `orbit.constants.constants.PredictedComponents`.
    title : str; optional
        title of the plot
    figsize : tuple; optional
        figsize pass through to `matplotlib.pyplot.figure()`
    path : str; optional
        path to save the figure
    fontsize : int; optional
        fontsize of the title
    is_visible : boolean
        whether we want to show the plot. If called from unittest, is_visible might = False.
   Returns
    -------
        matplotlib axes object
    """

    _predicted_df = predicted_df.copy()
    _predicted_df[date_col] = pd.to_datetime(_predicted_df[date_col])
    if plot_components is None:
        plot_components = [PredictionKeys.TREND.value,
                           PredictionKeys.SEASONALITY.value,
                           PredictionKeys.REGRESSION.value]

    plot_components = [p for p in plot_components if p in _predicted_df.columns.tolist()]
    n_panels = len(plot_components)
    if not figsize:
        figsize = (16, 8)

    if not fontsize:
        fontsize = 16

    if prediction_percentiles is None:
        _pred_percentiles = [5, 95]
    else:
        _pred_percentiles = prediction_percentiles

    if len(_pred_percentiles) != 2:
        raise ValueError("prediction_percentiles has to be None or a list with length=2.")

    fig, axes = plt.subplots(n_panels, 1, facecolor='w', figsize=figsize)
    for ax, comp in zip(axes, plot_components):
        y = predicted_df[comp].values
        ax.plot(_predicted_df[date_col], y, marker=None, color='#12939A')
        confid_cols = ["{}_{}".format(comp, _pred_percentiles[0]), "{}_{}".format(comp, _pred_percentiles[1])]
        if set(confid_cols).issubset(predicted_df.columns):
            ax.fill_between(_predicted_df[date_col].values,
                            _predicted_df[confid_cols[0]],
                            _predicted_df[confid_cols[1]],
                            facecolor='#42999E', alpha=0.5)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.5)
        ax.set_title(comp, fontsize=fontsize)
    plt.suptitle(title, fontsize=fontsize)
    fig.tight_layout()

    if path:
        plt.savefig(path)
    if is_visible:
        plt.show()
    else:
        plt.close()

    return axes


def metric_horizon_barplot(df, model_col='model', pred_horizon_col='pred_horizon',
                           metric_col='smape', bar_width=0.1, path=None,
                           figsize=None, fontsize=None, is_visible=False):
    if not figsize:
        figsize = [20, 6]

    if not fontsize:
        fontsize = 10

    plt.rcParams['figure.figsize'] = figsize
    models = df[model_col].unique()
    metric_horizons = df[pred_horizon_col].unique()
    n_models = len(models)
    palette = sns.color_palette("colorblind", n_models)

    # set height of bar
    bars = list()
    for m in models:
        bars.append(list(df[df[model_col] == m][metric_col]))

    # set position of bar on X axis
    r = list()
    r.append(np.arange(len(bars[0])))
    for idx in range(n_models - 1):
        r.append([x + bar_width for x in r[idx]])

    # make the plot
    for idx in range(n_models):
        plt.bar(r[idx], bars[idx], color=palette[idx], width=bar_width, edgecolor='white',
                label=models[idx])

    # add xticks on the middle of the group bars
    plt.xlabel('predict-horizon', fontweight='bold')
    plt.xticks([x + bar_width for x in range(len(bars[0]))], metric_horizons)

    # create legend & show graphic
    plt.legend()
    plt.title("Model Comparison with {}".format(metric_col), fontsize=fontsize)

    if path:
        plt.savefig(path)

    if is_visible:
        plt.show()
    else:
        plt.close()


def plot_posterior_params(mod, kind='density', n_bins=20, ci_level=.95,
                          pair_type='scatter', figsize=None, path=None, fontsize=None,
                          incl_trend_params=False, incl_smooth_params=False, is_visible=True):
    """ Data Viz for posterior samples

    Params
    ------
    mod : orbit model object
    kind : str, {'density', 'trace', 'pair'}
        which kind of plot to be made.
    n_bins : int; default 20
        number of bin, used in the histogram plotting
    ci_level : float, between 0 and 1
        confidence interval level
    pair_type : str, {'scatter', 'reg'}
        dot plotting type for off-diagonal plots in pair plot
    figsize : tuple; optional
        figure size
    path : str; optional
        dir path to save the chart
    fontsize: int; optional
        fontsize of the title
    incl_trend_params : bool
        if plot trend parameters; default False
    incl_smooth_params : bool
        if plot smoothing parameters; default False
    is_visible : boolean
        whether we want to show the plot. If called from unittest, is_visible might = False.

    Returns
    -------
        matplotlib axes object
    """
    if 'orbit' not in str(mod.__class__):
        raise Exception("This plotting utility works for orbit model object only.")
    # if mod.infer_method != 'mcmc':
    #     raise Exception("This plotting utility works for mcmc inference only.")
    if kind not in ['density', 'trace', 'pair']:
        raise Exception("kind must be one of 'density', 'trace', or 'pair'.")

    posterior_samples = deepcopy(mod._posterior_samples)

    if len(mod._regressor_col) > 0:
        for i, regressor in enumerate(mod._regressor_col):
            posterior_samples[regressor] = posterior_samples['beta'][:,i]

    params_ = mod._regressor_col + ['obs_sigma']

    if incl_trend_params:
        # trend params in LGT or DLT
        params_ += ['gt_pow', 'lt_coef', 'gt_coef', 'gb', 'gl']
    if incl_smooth_params:
        params_ += ['lev_sm', 'slp_sm', 'sea_sm']

    params_ = [x for x in params_ if x in posterior_samples.keys()]

    if not figsize:
        figsize = (8, 2 * len(params_))

    if not fontsize:
        fontsize = 10

    def _density_plot(posterior_samples, n_bins=20, ci_level=.95, figsize=None):

        fig, axes = plt.subplots(len(params_), 1, squeeze=True, figsize=figsize)
        for i, param in enumerate(params_):
            samples = posterior_samples[param]
            mean = np.mean(samples)
            median = np.median(samples)
            cred_min, cred_max = np.percentile(samples, 100 * (1 - ci_level)/2), \
                                 np.percentile(samples, 100 * (1 + ci_level)/2)

            sns.distplot(samples, bins=n_bins, kde_kws={'shade': True}, ax=axes[i], norm_hist=False)
            # sns.kdeplot(samples, shade=True, ax=axes[i])
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('density')
            # draw vertical lines
            axes[i].axvline(mean, color=QualitativePalette['PostQ'].value[0], lw=4, alpha=.5, label='mean')
            axes[i].axvline(median, color=QualitativePalette['PostQ'].value[1], lw=4, alpha=.5, label='median')
            axes[i].axvline(cred_min, linestyle='--', color='k', alpha=.5, label='95% CI')
            axes[i].axvline(cred_max, linestyle='--', color='k', alpha=.5)
            # axes[i].legend()

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.9))
        plt.suptitle('Histogram and Density of Posterior Samples', fontsize=fontsize)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return axes

    def _trace_plot(posterior_samples, ci_level=.95, figsize=None):

        fig, axes = plt.subplots(len(params_), 1, squeeze=True, figsize=figsize)
        for i, param in enumerate(params_):
            samples = posterior_samples[param]
            # chain order is preserved in the posterior samples
            chained_samples = np.array_split(samples, mod.estimator.chains)
            # mean = np.mean(samples)
            # median = np.median(samples)
            # cred_min, cred_max = np.percentile(samples, 100 * (1 - ci_level)/2), \
            #                         np.percentile(samples, 100 * (1 + ci_level)/2)

            for k in range(mod.estimator.chains):
                axes[i].plot(chained_samples[k], lw=1, alpha=.5, label=f'chain {k+1}')
            axes[i].set_ylabel(param)
            # axes[i].legend()
            # draw horizontal lines
            # axes[i].axhline(mean, color=QualitativePalette['PostQ'].value[0], lw=2, alpha=.5, label='mean')
            # axes[i].axhline(median, color=QualitativePalette['PostQ'].value[1], lw=2, alpha=.5, label='median')
            # axes[i].axhline(cred_min, linestyle='--', color='k', alpha=.5, label='95% CI')
            # axes[i].axhline(cred_max, linestyle='--', color='k', alpha=.5)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.9))
        plt.suptitle('Trace of Posterior Samples', fontsize=fontsize)
        plt.xlabel('draw')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return axes

    def _pair_plot(posterior_samples, pair_type='scatter', n_bins=20):
        samples_df = pd.DataFrame({key: posterior_samples[key].flatten() for key in params_})

        fig = sns.pairplot(samples_df, kind=pair_type, diag_kws=dict(bins=n_bins))
        fig.fig.suptitle("Pair Plot", fontsize=fontsize)
        fig.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    if kind == 'density':
        axes = _density_plot(posterior_samples, n_bins=n_bins, ci_level=ci_level, figsize=figsize)
    elif kind == 'trace':
        axes = _trace_plot(posterior_samples, ci_level=ci_level, figsize=figsize)
    elif kind == 'pair':
        axes = _pair_plot(posterior_samples, pair_type=pair_type, n_bins=n_bins)

    if path:
        plt.savefig(path)

    if is_visible:
        plt.show()
    else:
        plt.close()

    return axes





def get_arviz_plot_dict(mod,
                        incl_noise_params=False,
                        incl_trend_params=False,
                        incl_smooth_params=False):
    """ This is a utility to prepare the plotting dictionary data for package arviz.
    arviz will interpret each key as the name of a different random variable.
    Each array should have shape (chains, draws, *shape).

    See Also
    --------
    https://arviz-devs.github.io/arviz/index.html
    """
    if not ('MCMC' in str(mod.estimator) or 'VI' in str(mod.estimator)):
        raise Exception("This utility works for model object with MCMC or VI inference only.")

    posterior_samples = mod.get_posterior_samples()
    if len(mod._regressor_col) > 0:
        for i, regressor in enumerate(mod._regressor_col):
            posterior_samples[regressor] = posterior_samples['beta'][:,i]
    params_ = mod._regressor_col

    if incl_noise_params:
        params_ += ['obs_sigma']
    if incl_trend_params:
        params_ += ['gt_pow', 'lt_coef', 'gt_coef', 'gb', 'gl']
    if incl_smooth_params:
        params_ += ['lev_sm', 'slp_sm', 'sea_sm']

    params_ = [x for x in params_ if x in posterior_samples.keys()]

    for key in posterior_samples.copy().keys():
        if key not in params_: del posterior_samples[key]

    for key, val in posterior_samples.items():
        posterior_samples[key] = val.reshape((mod.estimator.chains,
                                              mod.estimator._num_sample_per_chain,
                                              *val.shape[1:]))

    return posterior_samples


def plot_param_diagnostics(mod, incl_noise_params=False, incl_trend_params=False, incl_smooth_params=False,
                           which='trace', **kwargs):
    """
    Parameters
    -----------
    mod : orbit model object
    which : str, {'density', 'trace', 'pair', 'autocorr', 'posterior', 'forest'}
    incl_noise_params : bool
        if plot noise parameters; default False
    incl_trend_params : bool
        if plot trend parameters; default False
    incl_smooth_params : bool
        if plot smoothing parameters; default False
    **kwargs :
        other parameters passed to arviz functions

    Returns
    -------
        matplotlib axes object
    """
    posterior_samples = get_arviz_plot_dict(mod,
                                            incl_noise_params=incl_noise_params,
                                            incl_trend_params=incl_trend_params,
                                            incl_smooth_params=incl_smooth_params)

    if which == "trace":
        axes = az.plot_trace(posterior_samples, **kwargs)
    elif which == "density":
        axes = az.plot_density(posterior_samples, **kwargs)
    elif which == "posterior":
        axes = az.plot_posterior(posterior_samples, **kwargs)
    elif which == "pair":
        axes = az.plot_pair(posterior_samples, **kwargs)
    elif which == "autocorr":
        axes = az.plot_autocorr(posterior_samples, **kwargs)
    elif which == "forest":
        axes = az.plot_forest(posterior_samples, **kwargs)
    else:
        raise Exception("please use one of 'trace', 'density', 'posterior', 'pair', 'autocorr', 'forest' for kind.")

    return axes
