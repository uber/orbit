import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import math

from orbit.constants import palette
from orbit.utils.plot import orbit_style_decorator

pd.options.mode.chained_assignment = None


@orbit_style_decorator
def ts_heatmap(
    df,
    date_col,
    value_col,
    seasonal_interval,
    fig_width=8,
    fig_height=8,
    normalization=False,
    path=None,
    palette=palette.OrbitColorMap.BLUE_GRADIENT.value,
):
    """this function takes a time series dataframe and plot a time series heatmap with month on the y axis and
    year on the x axis
    Parameters
    -----------
    df : pandas data frame
        input df
    date_col : str
        the name of the date column
    value_col : str
        the name of the value
    seasonal_interval: int
        the desired seasonal_interval, this is used for the y axis
    fig_width : int, optional
        adjust width of the chart
    fig_height : int, optional,
        adjust height of the chart
    normalization : bool, optional
        normalize using mean and std
    path : str
        path to save the figure
    palette : str, optional
        color palette

    Returns
    -------
        a time series heatmap chart, with year as x axis, and desired seasonal interval as y axis
    """
    df = df[[date_col, value_col]]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col]).reset_index()
    length = df.shape[0]
    ses_interval = []
    for i in range(math.ceil(length / seasonal_interval)):
        ses_interval += [i] * seasonal_interval
    df["seasonal_interval"] = ses_interval[0:length]
    df["y_axis_interval"] = df["index"] % seasonal_interval

    # df['month'] = df[date_col].dt.month
    df_pivot = df.pivot_table(
        index="y_axis_interval", columns="seasonal_interval", values=value_col
    ).sort_index(ascending=False)
    if normalization:
        df_pivot = (df_pivot - df_pivot.mean()) / df_pivot.std()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax = sns.heatmap(df_pivot, cmap=palette, linewidths=0.8)

    if normalization:
        ax.set_title(f"{value_col} Time Series Heatmap Normalized Index")

    ax.set_title(f"{value_col} (Mean) Time Series Heatmap")

    if path:
        fig.savefig(path)

    plt.tight_layout()
    return ax, df, df_pivot


@orbit_style_decorator
def correlation_heatmap(
    df,
    var_list,
    fig_width=8,
    fig_height=8,
    path=None,
    fmt=".1g",
    palette=palette.OrbitColorMap.BLUE_GRADIENT.value,
):
    """This function takes a list of variables and return a heatmap of pairwise correlation. The columns with
    all zero values will not be plotted.
    Parameters
    -----------
    df : pandas data frame
        input df
    var_list : list
        list of the variable names
    fig_width : int, optional
        adjust width of the chart
    fig_height : int, optional,
        adjust height of the chart
    path : str
        path to save the figure
    palette : str, optional
        color palette

    Returns
    -------
    one correlation heatmap chart
    """
    # filter out all zero columns
    non_zero_varlist = [i for i in var_list if i not in df.columns[(df == 0).all()]]
    df = df[non_zero_varlist].corr()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax = sns.heatmap(df, cmap=palette, annot=True, fmt=fmt, linewidths=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")

    ax.set_title("Correlation Heatmap")

    if path:
        fig.savefig(path)

    return ax


@orbit_style_decorator
def dual_axis_ts_plot(
    df,
    var1,
    var2,
    date_col,
    fig_width=25,
    fig_height=6,
    path=None,
    color1=palette.OrbitPalette.BLACK.value,
    color2=palette.OrbitPalette.BLUE.value,
):
    """This function plots two time series variables on two y axis. This is handy for comparison of two variables. The dual
    y axis will set on two different scales if the two variables are very different in terms of volume.
    Parameters
    -----------
    df : pandas data frame
        input df
    var1 : str
        name of the first variable
    var2 : str
        name of the second variable
    date_col : str
        date column name
    fig_width : int, optional
        adjust width of the chart
    fig_height : int, optional,
        adjust height of the chart
    path : str
        path to save the figure
    color1: str
        color for var1
    color2: str
        color for var2

    Returns
    -------
    one time series plot with dual y axis
    """
    df[date_col] = pd.to_datetime(df[date_col])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.lineplot(data=df, x=date_col, y=var1, label=var1, color=color1)
    ax.set_ylabel(f"{var1}", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax2 = ax.twinx()
    sns.lineplot(data=df, x=date_col, y=var2, ax=ax2, color=color2, label=var2)
    ax2.set_ylabel(f"{var2}", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax.set_title(f"{var1} vs {var2}")
    # ax.grid(False)
    ax2.grid(False)

    if path:
        fig.savefig(path)

    return ax


@orbit_style_decorator
def wrap_plot_ts(
    df,
    date_col,
    var_list,
    col_wrap=3,
    height=4,
    aspect=2,
    palettes=palette.OrbitPalette.BLUE.value,
):
    """This function plots a panel of time series plots.
    Parameters
    -----------
    df : pandas data frame
        input df
    date_col : str
        date column name
    var_list : list
        list of variable names

    Returns
    -------
    seaborn facet grid axes object
    """
    non_zero_varlist = [i for i in var_list if i not in df.columns[(df == 0).all()]]
    df = df[non_zero_varlist]
    df_long = df.melt(id_vars=[date_col])
    colors = [palettes]
    custom_palette = sns.set_palette(sns.color_palette(colors))

    ax = sns.relplot(
        x=date_col,
        y="value",
        col="variable",
        height=height,
        aspect=aspect,
        col_wrap=col_wrap,
        kind="line",
        data=df_long,
        palette=custom_palette,
        facet_kws={"sharey": False, "sharex": False},
    )
    ax.set_xticklabels(rotation=45)
    plt.tight_layout()

    return ax


# def weekly_trend_decomposition(df, var, date_col):
#     """This function presents variable weekly trend after removing weekly seasonality.
#     Parameters
#     -----------
#     df : pandas data frame
#         input df
#     var : str
#         name of the variable
#     date_col : str
#         the name of the date column

#     Returns
#     -------
#     three subplots with Trend, Residual and Weekly Seasonality
#     """
#     df_dt = deepcopy(df)
#     df_dt.index = pd.to_datetime(df_dt[date_col])
#     res_weely = seasonal_decompose(df_dt[[var]], period=7, model='multiplicative')
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
#     res_weely.trend.plot(ax=ax1)
#     res_weely.resid.plot(ax=ax2)
#     res_weely.seasonal.plot(ax=ax3)
#     ax1.set_title("Removing Weekly Seasonality")
#     ax1.set_ylabel("Trend")
#     ax2.set_ylabel("Residual")
#     ax3.set_ylabel("Weekly Seasonality")


# def year_over_year_ts_plot(df, date_col, outcome, event_df, event_col,
#                            fig_width=30, fig_height=12, path=None):
#     """This chart plots outcome time series year over year against events to understand seasonality and trend, and how
#     events impact the pattern of outcome. The chart contains two subplots: 1) year over year time series plot of
#     Parameters
#     -----------
#     df : pandas data frame
#         input df
#     date_col : str
#         the name of the date column
#     outcome : str
#         the name of variable to be inspected
#     event_df : pandas data frame
#         event input data
#     event_col : str
#         name of the event column
#     fig_width : int, optional
#         adjust width of the chart
#     fig_height : int, optional,
#         adjust height of the chart
#     path : str
#         path to save the figure

#     Returns
#     -------
#     one chart with two subplots
#     """

#     df[date_col] = pd.to_datetime(df[date_col])
#     df['year'] = df[date_col].dt.year
#     df['month'] = df[date_col].dt.month
#     df['day'] = df[date_col].dt.day
#     df['day_of_month'] = df.month.astype(str) + '_' + df.day.astype(str)

#     event_df[date_col] = pd.to_datetime(event_df[date_col])
#     event_df['year'] = event_df[date_col].dt.year
#     event_df['month'] = event_df[date_col].dt.month
#     event_df['day'] = event_df[date_col].dt.day
#     event_df['day_of_month'] = event_df.month.astype(str) + '_' + event_df.day.astype(str)
#     event_df = event_df[event_df[date_col] <= df[date_col].max()]
#     color = iter(cm.rainbow(np.linspace(0, 1, event_df[event_col].nunique())))

#     fig, ax = plt.subplots(2, 1, figsize=(fig_width, fig_height))
#     # the first plot year over year time series vs events
#     for i in df.year.unique():
#         sns.lineplot(data=df[df['year'] == i], x='day_of_month', y=outcome, ax=ax[0], label=i)
#     ax[0].set_xticks(np.arange(0, 366, 14))
#     ax[0].title.set_text(f'{outcome} Year Over Year Plot')

#     h_list = event_df[event_col].unique()
#     for h in h_list:
#         h_df = event_df[event_df[event_col] == h]
#         c = next(color)
#         for i in h_df.day_of_month.unique():
#             ax[0].axvline(x=i, linestyle='--', color=c, label=h)
#         for d in h_df[date_col].unique():
#             ax[1].axvline(x=d, linestyle='--', color=c, label=h)
#     handles, labels = ax[0].get_legend_handles_labels()
#     handle_list, label_list = [], []
#     for handle, label in zip(handles, labels):
#         if label not in label_list:
#             handle_list.append(handle)
#             label_list.append(label)
#     ax[0].legend(handle_list, label_list, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
#     # the second plot: outcome vs event
#     sns.lineplot(data=df, x=date_col, y=outcome, ax=ax[1])
#     ax[1].legend(handle_list, label_list, loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
#     ax[1].title.set_text(f'{outcome} Time Series with Events Plot')

#     if path:
#         fig.savefig(path)

#     return ax
