import numpy as np
import pandas as pd


def prepend_date_column(predicted_df, input_df, date_col):
    """Prepends date column from `input_df` to `predicted_df`
    Parameters
    ----------
    predicted_df : pd.DataFrame
    input_df : pd.DataFrame
    date_col : str
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


def compute_percentiles(predictions_dict, percentiles):
    """Compute percentiles of dictionary of arrays.  Return results as dataframe.
    Parameters
    ----------
    predictions_dict : dict
        a dictionary where keys will be the output columns of a dataframe and
        values are a 2d numpy array of shape (`num_samples`, prediction df length)
    percentiles : list
        A sorted list of one or three percentile(s) which will be used to aggregate lower, mid and upper values
    Returns
    -------
    pd.DataFrame
        The aggregated across mcmc samples with columns for `50` aka median
        and all other percentiles specified in `percentiles`.
    """

    computed_dict = {}
    run_check = False
    prev_shape = None
    for k, v in predictions_dict.items():
        curr_shape = v.shape
        if curr_shape != 2:
            raise ValueError("Input predictions_dict requires 2 dimensions: (number of samples, prediction length)."
                             "Please revise input.")
        if run_check:
            if curr_shape[1] != prev_shape[1]:
                raise ValueError("Input components length of are inconsistent. Please revise input.")
        # run prediction consistency after first iteration
        run_check = True
        prev_shape = curr_shape
        computed_array = np.percentile(v, percentiles, axis=0)
        columns = [k + "_" + str(p) if p != 50 else k for p in percentiles]
        computed_dict[k] = pd.DataFrame(computed_array.T, columns=columns)

    output_df = pd.concat(computed_dict, axis=1)
    output_df.columns = output_df.columns.droplevel()
    return output_df
