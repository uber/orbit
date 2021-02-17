import numpy as np
import pandas as pd


def prepend_date_column(predicted_df, input_df, date_col):
    """Prepends date column from `input_df` to `predicted_df`
    Parameters
    ----------
    predicted_df: pd.DataFrame
    input_df: pd.DataFrame
    date_col: str
    Returns
    -------
    pd.DataFrame
    """

    other_cols = list(predicted_df.columns)

    # add date column
    predicted_df[date_col] = input_df[date_col].reset_index(drop=True)

    # re-order columns so date is first
    col_order = [date_col] + other_cols
    predicted_df = predicted_df[col_order]

    return predicted_df


def aggregate_predictions(predictions_dict, percentiles):
    """Aggregates the mcmc prediction to a point estimate
    Parameters
    ----------
    predictions_dict: dict
        a dictionary where keys will be the output columns of a dataframe and
        values are a 2d numpy array of shape (`num_samples`, prediction df length)
    percentiles: list
        A sorted list of one or three percentile(s) which will be used to aggregate lower, mid and upper values
    Returns
    -------
    pd.DataFrame
        The aggregated across mcmc samples with columns for `50` aka median
        and all other percentiles specified in `percentiles`.
    """

    aggregated_dict = {}

    for k, v in predictions_dict.items():
        aggregated_array = np.percentile(v, percentiles, axis=0)
        columns = [k + "_" + str(p) if p != 50 else k for p in percentiles]
        aggregated_dict[k] = pd.DataFrame(aggregated_array.T, columns=columns)

    aggregated_df = pd.concat(aggregated_dict, axis=1)
    aggregated_df.columns = aggregated_df.columns.droplevel()
    return aggregated_df
