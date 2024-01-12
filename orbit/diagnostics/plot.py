# the following lines are added to fix unit test error
# or else the following line will give the following error
# TclError: no display name and no $DISPLAY environment variable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os
import pkg_resources
import statsmodels.api as sm
from scipy import stats

from ..constants.constants import PredictionKeys
from orbit.utils.general import is_empty_dataframe, is_ordered_datetime
from ..constants.constants import BacktestFitKeys
from ..constants.palette import PredictionPaletteClassic as PredPal
from orbit.constants import palette
from orbit.diagnostics.metrics import smape
from orbit.utils.plot import orbit_style_decorator
from ..exceptions import PlotException


import logging

logger = logging.getLogger("orbit")


@orbit_style_decorator
def plot_predicted_data(
    training_actual_df,
    predicted_df,
    date_col,
    actual_col,
    pred_col=PredictionKeys.PREDICTION.value,
    prediction_percentiles=None,
    title="",
    test_actual_df=None,
    is_visible=True,
    figsize=None,
    path=None,
    fontsize=None,
    line_plot=False,
    markersize=50,
    lw=2,
    linestyle="-",
):
    """plot training actual response together with predicted data; if actual response of predicted
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
        raise ValueError(
            "prediction_percentiles has to be None or a list with length=2."
        )

    confid_cols = [
        "prediction_{}".format(_pred_percentiles[0]),
        "prediction_{}".format(_pred_percentiles[1]),
    ]

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

    fig, ax = plt.subplots(facecolor="w", figsize=figsize)

    if line_plot:
        ax.plot(
            _training_actual_df[date_col].values,
            _training_actual_df[actual_col].values,
            marker=None,
            color=PredPal.ACTUAL_OBS.value,
            lw=lw,
            label="train response",
            linestyle=linestyle,
        )
    else:
        ax.scatter(
            _training_actual_df[date_col].values,
            _training_actual_df[actual_col].values,
            marker=".",
            color=PredPal.ACTUAL_OBS.value,
            alpha=0.8,
            s=markersize,
            label="train response",
        )

    ax.plot(
        _predicted_df[date_col].values,
        _predicted_df[pred_col].values,
        marker=None,
        color=PredPal.PREDICTION_LINE.value,
        lw=lw,
        label=PredictionKeys.PREDICTION.value,
        linestyle=linestyle,
    )

    # vertical line separate training and prediction
    if _training_actual_df[date_col].values[-1] < _predicted_df[date_col].values[-1]:
        ax.axvline(
            x=_training_actual_df[date_col].values[-1],
            color=PredPal.HOLDOUT_VERTICAL_LINE.value,
            alpha=0.5,
            linestyle="--",
        )

    if test_actual_df is not None:
        test_actual_df = test_actual_df.copy()
        test_actual_df[date_col] = pd.to_datetime(test_actual_df[date_col])
        if line_plot:
            ax.plot(
                test_actual_df[date_col].values,
                test_actual_df[actual_col].values,
                marker=None,
                color=PredPal.TEST_OBS.value,
                lw=lw,
                label="train response",
                linestyle=linestyle,
            )
        else:
            ax.scatter(
                test_actual_df[date_col].values,
                test_actual_df[actual_col].values,
                marker=".",
                color=PredPal.TEST_OBS.value,
                s=markersize,
                label="test response",
            )

    # prediction intervals
    if plot_confid:
        ax.fill_between(
            _predicted_df[date_col].values,
            _predicted_df[confid_cols[0]],
            _predicted_df[confid_cols[1]],
            facecolor=PredPal.PREDICTION_INTERVAL.value,
            alpha=0.3,
        )

    ax.set_title(title, fontsize=fontsize)
    # ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.5) --comment out since we have orbit style
    ax.legend()
    if path:
        fig.savefig(path)
    if is_visible:
        plt.show()
    else:
        plt.close()

    return ax


@orbit_style_decorator
def plot_predicted_components(
    predicted_df,
    date_col,
    prediction_percentiles=None,
    plot_components=None,
    title="",
    figsize=None,
    path=None,
    fontsize=None,
    is_visible=True,
):
    """Plot predicted components with the data frame of decomposed prediction where components
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
        plot_components = [
            PredictionKeys.TREND.value,
            PredictionKeys.SEASONALITY.value,
            PredictionKeys.REGRESSION.value,
        ]

    plot_components = [
        p for p in plot_components if p in _predicted_df.columns.tolist()
    ]
    nrows = len(plot_components)
    if not figsize:
        figsize = (16, 8)

    if not fontsize:
        fontsize = 16

    if prediction_percentiles is None:
        _pred_percentiles = [5, 95]
    else:
        _pred_percentiles = prediction_percentiles

    if len(_pred_percentiles) != 2:
        raise ValueError(
            "prediction_percentiles has to be None or a list with length=2."
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    for ax, comp in zip(axes, plot_components):
        y = predicted_df[comp].values
        ax.plot(
            _predicted_df[date_col],
            y,
            marker=None,
            color=PredPal.PREDICTION_INTERVAL.value,
        )
        confid_cols = [
            "{}_{}".format(comp, _pred_percentiles[0]),
            "{}_{}".format(comp, _pred_percentiles[1]),
        ]
        if set(confid_cols).issubset(predicted_df.columns):
            ax.fill_between(
                _predicted_df[date_col].values,
                _predicted_df[confid_cols[0]],
                _predicted_df[confid_cols[1]],
                facecolor=PredPal.PREDICTION_INTERVAL.value,
                alpha=0.3,
            )
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


@orbit_style_decorator
def plot_bt_predictions(
    bt_pred_df,
    metrics=smape,
    split_key_list=None,
    ncol=2,
    figsize=None,
    include_vline=True,
    title="",
    fontsize=20,
    path=None,
    is_visible=True,
):
    """function to plot and visualize the prediction results from back testing.

    bt_pred_df : data frame
        the output of `orbit.diagnostics.backtest.BackTester.fit_predict()`, which includes the actuals/predictions
        for all the splits
    metrics : callable
        the metric function
    split_key_list: list; default None
        with given model, which split keys to plot. If None, all the splits will be plotted
    ncol : int
        number of columns of the panel; number of rows will be decided accordingly
    figsize : tuple
        figure size
    include_vline : bool
        if plotting the vertical line to cut the in-sample and out-of-sample predictions for each split
    title : str
        title of the plot
    fontsize: int; optional
        fontsize of the title
    path : string
        path to save the figure
    is_visible : bool
        if displaying the figure
    """
    if figsize is None:
        figsize = (16, 8)

    metric_vals = bt_pred_df.groupby(BacktestFitKeys.SPLIT_KEY.value).apply(
        lambda x: metrics(
            x[~x[BacktestFitKeys.TRAIN_FLAG.value]][BacktestFitKeys.ACTUAL.value],
            x[~x[BacktestFitKeys.TRAIN_FLAG.value]][BacktestFitKeys.PREDICTED.value],
        )
    )

    if split_key_list is None:
        split_key_list_ = bt_pred_df[BacktestFitKeys.SPLIT_KEY.value].unique()
    else:
        split_key_list_ = split_key_list

    num_splits = len(split_key_list_)
    nrow = math.ceil(num_splits / ncol)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=figsize,
        squeeze=False,
        facecolor="w",
        constrained_layout=False,
    )

    for idx, split_key in enumerate(split_key_list_):
        row_idx = idx // ncol
        col_idx = idx % ncol
        tmp = bt_pred_df[
            bt_pred_df[BacktestFitKeys.SPLIT_KEY.value] == split_key
        ].copy()
        axes[row_idx, col_idx].plot(
            tmp[BacktestFitKeys.DATE.value],
            tmp[BacktestFitKeys.PREDICTED.value],
            # linewidth=2,
            color=PredPal.PREDICTION_LINE.value,
        )
        axes[row_idx, col_idx].scatter(
            tmp[BacktestFitKeys.DATE.value],
            tmp[BacktestFitKeys.ACTUAL.value],
            label=BacktestFitKeys.ACTUAL.value,
            color=PredPal.ACTUAL_OBS.value,
            alpha=0.6,
            s=8,
        )
        # axes[row_idx, col_idx].grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.4)

        axes[row_idx, col_idx].set_title(
            label="split {}; {} {:.3f}".format(
                split_key, metrics.__name__, metric_vals[split_key]
            )
        )
        if include_vline:
            cutoff_date = tmp[~tmp[BacktestFitKeys.TRAIN_FLAG.value]][
                BacktestFitKeys.DATE.value
            ].min()
            axes[row_idx, col_idx].axvline(
                x=cutoff_date,
                linestyle="--",
                color=PredPal.HOLDOUT_VERTICAL_LINE.value,
                # linewidth=4,
                alpha=0.8,
            )

    plt.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    if is_visible:
        plt.show()
    else:
        plt.close()

    return axes


@orbit_style_decorator
def plot_bt_predictions2(
    bt_pred_df,
    metrics=smape,
    split_key_list=None,
    figsize=None,
    include_vline=True,
    title="",
    fontsize=20,
    markersize=50,
    lw=2,
    fig_dir=None,
    is_visible=True,
    fix_xylim=True,
    export_gif=False,
):
    """a different style backtest plot compare to `plot_bt_prediction` where it writes separate plot for each split;
    this is also used to produce an animation to summarize every split
    """
    if figsize is None:
        figsize = (16, 8)

    if fig_dir:
        if not os.path.isdir(fig_dir) or not os.path.exists(fig_dir):
            raise PlotException(
                "Invalid or non-existing directory use specified: {}.".format(
                    os.path.abspath(fig_dir)
                )
            )
        fig_paths = list()

    metric_vals = bt_pred_df.groupby(BacktestFitKeys.SPLIT_KEY.value).apply(
        lambda x: metrics(
            x[~x[BacktestFitKeys.TRAIN_FLAG.value]][BacktestFitKeys.ACTUAL.value],
            x[~x[BacktestFitKeys.TRAIN_FLAG.value]][BacktestFitKeys.PREDICTED.value],
        )
    )

    if split_key_list is None:
        split_key_list_ = bt_pred_df[BacktestFitKeys.SPLIT_KEY.value].unique()
    else:
        split_key_list_ = split_key_list

    if fix_xylim:
        all_values = np.concatenate(
            (
                bt_pred_df[BacktestFitKeys.ACTUAL.value].values,
                bt_pred_df[BacktestFitKeys.PREDICTED.value].values,
            )
        )
        ylim = (np.min(all_values) * 0.99, np.max(all_values) * 1.01)
        xlim = (
            bt_pred_df[BacktestFitKeys.DATE.value].values[0],
            bt_pred_df[BacktestFitKeys.DATE.value].values[-1],
        )

    for idx, split_key in enumerate(split_key_list_):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        tmp = bt_pred_df[
            bt_pred_df[BacktestFitKeys.SPLIT_KEY.value] == split_key
        ].copy()
        ax.plot(
            tmp[BacktestFitKeys.DATE.value],
            tmp[BacktestFitKeys.PREDICTED.value],
            color=PredPal.PREDICTION_LINE.value,
            lw=lw,
        )

        train_df = tmp.loc[tmp[BacktestFitKeys.TRAIN_FLAG.value], :]
        ax.scatter(
            train_df[BacktestFitKeys.DATE.value],
            train_df[BacktestFitKeys.ACTUAL.value],
            marker=".",
            color=PredPal.ACTUAL_OBS.value,
            alpha=0.8,
            s=markersize,
            label="train response",
        )

        test_df = tmp.loc[~tmp[BacktestFitKeys.TRAIN_FLAG.value], :]
        ax.scatter(
            test_df[BacktestFitKeys.DATE.value],
            test_df[BacktestFitKeys.ACTUAL.value],
            marker=".",
            color=PredPal.TEST_OBS.value,
            alpha=0.8,
            s=markersize,
            label="test response",
        )

        ax.set_title(
            label="split {}; {} {:.3f}".format(
                split_key, metrics.__name__, metric_vals[split_key]
            )
        )
        if include_vline:
            cutoff_date = tmp[~tmp[BacktestFitKeys.TRAIN_FLAG.value]][
                BacktestFitKeys.DATE.value
            ].min()
            ax.axvline(
                x=cutoff_date,
                linestyle="--",
                color=PredPal.HOLDOUT_VERTICAL_LINE.value,
                alpha=0.8,
            )
        if fix_xylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.legend()
        plt.suptitle(title, fontsize=fontsize)
        fig.tight_layout()

        if fig_dir:
            fig_path = "{}/splits_{}.png".format(fig_dir, idx)
            fig_paths.append(fig_path)
            fig.savefig(fig_path)
        if is_visible:
            plt.show()
        else:
            plt.close()

    if fig_dir and export_gif:
        package_name = "imageio"
        try:
            pkg_resources.get_distribution(package_name)
            import imageio

            with imageio.get_writer(
                "{}/orbit-backtest.gif".format(fig_dir), mode="I"
            ) as writer:
                for fig_path in fig_paths:
                    image = imageio.imread(fig_path)
                    writer.append_data(image)
        except pkg_resources.DistributionNotFound:
            logger.error(
                (
                    "{} not installed, which is necessary for gif animation".format(
                        package_name
                    )
                )
            )


# TODO: update palette
@orbit_style_decorator
def metric_horizon_barplot(
    df,
    model_col="model",
    pred_horizon_col="pred_horizon",
    metric_col="smape",
    bar_width=0.1,
    path=None,
    figsize=None,
    fontsize=None,
    is_visible=False,
):
    if not figsize:
        figsize = [20, 6]

    if not fontsize:
        fontsize = 10

    plt.rcParams["figure.figsize"] = figsize
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
        plt.bar(
            r[idx],
            bars[idx],
            color=palette[idx],
            width=bar_width,
            edgecolor="white",
            label=models[idx],
        )

    # add xticks on the middle of the group bars
    plt.xlabel("predict-horizon", fontweight="bold")
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


@orbit_style_decorator
def params_comparison_boxplot(
    data,
    var_names,
    model_names,
    color_list=sns.color_palette(),
    title="Params Comparison",
    fig_size=(10, 6),
    box_width=0.1,
    box_distance=0.2,
    showfliers=False,
):
    """compare the distribution of parameters from different models uisng a boxplot.
    Parameters:
        data : a list of dict with keys as the parameters of interest
        var_names : a list of strings, the labels of the parameters to compare
        model_names : a list of strings, the names of models to compare
        color_list : a list of strings, the color to use for differentiating models
        title : string
            the title of the chart
        fig_size : tuple
            figure size
        box_width : float
            width of the boxes in the boxplot
        box_distance : float
            the distance between each boxes in the boxplot
        showfliers  : boolean
            show outliers in the chart if set as True

    Returns:
        a boxplot comparing parameter distributions from different models side by side
    """

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    handles = []
    n_models = len(model_names)
    pos = []

    if n_models % 2 == 0:
        for n in range(1, int(n_models / 2) + 1):
            pos.append(round(box_distance * (-1) ** (n_models - 1) * n, 1))
            pos.append(round(box_distance * (-1) ** (n_models) * n, 1))

    else:
        for n in range(1, int((n_models - 1) / 2) + 1):
            pos.append(0)
            pos.append(round(box_distance * (-1) ** (n_models - 1) * n, 1))
            pos.append(round(box_distance * (-1) ** (n_models) * n, 1))

    pos = sorted(pos)

    for i in range(len(model_names)):
        plt_arr = []
        for var in var_names:
            plt_arr.append(data[i][var].flatten())
        plt_arr = np.vstack(plt_arr).T
        globals()[f"bp{i}"] = ax.boxplot(
            plt_arr,
            positions=np.arange(plt_arr.shape[1]) + pos[i],
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor=color_list[i]),
            medianprops=dict(color="black"),
            showfliers=showfliers,
        )
        handles.append(globals()[f"bp{i}"]["boxes"][0])

    plt.xticks(np.arange(len(var_names)), var_names)
    ax.legend(handles, model_names)
    plt.xlabel("params")
    plt.ylabel("value")
    plt.title(title)

    return ax


@orbit_style_decorator
def residual_diagnostic_plot(
    df,
    dist="norm",
    date_col="week",
    residual_col="residual",
    fitted_col="prediction",
    sparams=None,
):
    """
    Parameters
    ----------

    df : pd.DataFrame
    dist : str
    date_col : str
        column name of date
    residual_col : str
        column name of residual
    fitted_col: str
        column name of fitted value from model
    sparams : float or list
        extra parameters used in distribution such as t-dist

    Notes
    -----
    1. residual by time
    2. residual vs fitted
    3. residual histogram with vertical line as mean
    4. residuals qq plot
    5. residual ACF
    6. residual PACF
    """
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))

    # plot 1 residual by time
    sns.lineplot(
        x=date_col,
        y=residual_col,
        data=df,
        ax=ax[0, 0],
        color=palette.OrbitPalette.BLUE.value,
        alpha=0.8,
        label="residual",
    )
    ax[0, 0].set_title("Residual by Time")
    ax[0, 0].legend()

    # plot 2 residual vs fitted
    sns.scatterplot(
        x=fitted_col,
        y=residual_col,
        data=df,
        ax=ax[0, 1],
        color=palette.OrbitPalette.BLUE.value,
        alpha=0.8,
        label="residual",
    )
    ax[0, 1].axhline(
        y=0,
        linestyle="--",
        color=palette.OrbitPalette.BLACK.value,
        alpha=0.5,
        label="0",
    )
    ax[0, 1].set_title("Residual vs Fitted")
    ax[0, 1].set_xlabel("fitted")
    ax[0, 1].legend()

    # plot 3 residual histogram with vertical line as mean
    sns.histplot(
        df[residual_col].values,
        kde=True,
        ax=ax[1, 0],
        color=palette.OrbitPalette.BLUE.value,
        label="residual",
        edgecolor="white",
        alpha=0.5,
        facecolor=palette.OrbitPalette.BLUE.value,
    )
    ax[1, 0].set_title("Residual Distribution")
    ax[1, 0].axvline(
        df[residual_col].mean(),
        color=palette.OrbitPalette.ORANGE.value,
        linestyle="--",
        alpha=0.9,
        label="residual mean",
    )
    ax[1, 0].set_ylabel("density")
    ax[1, 0].legend()

    # plot 4 residual qq plot
    if dist == "norm":
        _ = stats.probplot(df[residual_col].values, dist="norm", plot=ax[1, 1])
    elif dist == "t-dist":
        # t-dist qq-plot
        _ = stats.probplot(
            df[residual_col].values, dist=stats.t, sparams=sparams, plot=ax[1, 1]
        )

    # plot 5 residual ACF
    sm.graphics.tsa.plot_acf(
        df[residual_col].values,
        ax=ax[2, 0],
        title="Residual ACF",
        color=palette.OrbitPalette.BLUE.value,
    )
    ax[2, 0].set_xlabel("lag")
    ax[2, 0].set_ylabel("acf")

    # plot 6 residual PACF
    sm.graphics.tsa.plot_pacf(
        df[residual_col].values,
        ax=ax[2, 1],
        title="Residual PACF",
        color=palette.OrbitPalette.BLUE.value,
    )
    ax[2, 1].set_xlabel("lag")
    ax[2, 1].set_ylabel("pacf")
    fig.tight_layout()
