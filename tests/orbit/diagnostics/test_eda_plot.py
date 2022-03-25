import pytest
import numpy as np

from orbit.eda import eda_plot


def test_eda_plot(iclaims_training_data):
    df = iclaims_training_data
    df["claims"] = np.log(df["claims"])

    # test plotting
    _ = eda_plot.ts_heatmap(
        df=df,
        date_col="week",
        value_col="claims",
        seasonal_interval=52,
        normalization=True,
    )

    var_list = ["trend.unemploy", "trend.filling", "trend.job"]
    _ = eda_plot.correlation_heatmap(df, var_list=var_list)

    _ = eda_plot.dual_axis_ts_plot(
        df=df, var1="trend.unemploy", var2="claims", date_col="week"
    )

    df[["week"] + var_list].melt(id_vars=["week"])
    _ = eda_plot.wrap_plot_ts(df, "week", ["week"] + var_list)
