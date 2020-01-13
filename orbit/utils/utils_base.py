from collections import defaultdict
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# def compile_stan_file_to_pickle_file(stan_file_path):
#     """
#     we generate a pickle file from the given stan file.
#     The pickle file is compiled from the stan file.
#
#     Parameters
#     ----------
#     stan_file_path: str
#         The path string pointing to stan file;
#     -------
#
#     """
#     pickle_file_path = re.sub('.stan$', ".pkl", stan_file_path, 1)
#     compiled_pickle_file_from_stan = pystan.StanModel(file=stan_file_path)
#     with open(pickle_file_path, 'wb') as f:
#         pickle.dump(compiled_pickle_file_from_stan, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_date_interval(first_date, second_date):
    """

    Parameters
    ----------
    first_date: str
        earlier date, with format "%Y-%m-%d"
    second_date: str
        later date, with format "%Y-%m-%d"

    Returns: either 'day', 'week' or 'month'
    -------

    """
    diff = datetime.datetime.strptime(second_date, "%Y-%m-%d") - datetime.datetime.strptime(
        first_date, "%Y-%m-%d",
    )
    if diff == datetime.timedelta(days=1):
        date_interval = 'day'
    elif diff == datetime.timedelta(days=7):
        date_interval = 'week'
    elif datetime.timedelta(days=28) <= diff <= datetime.timedelta(days=31):
        date_interval = 'month'
    else:
        raise ValueError("Date interval not well defined, only support day, week and month.")

    return date_interval


def get_num_of_intervals_between_two_dates(first_date, second_date, date_interval):
    """
    Calculate the number of days/weeks/months between the given two dates

    Parameters
    ----------
    first_date: str or datetime object
        given first date
    second_date: str or datetime object
        given second date
    date_interval: str
        can only be day, week or month

    Returns
    -------
        int
        The number of days/weeks/months between the first date and second date.

    """
    first_date_obj = datetime.datetime.strptime(first_date, "%Y-%m-%d") \
        if isinstance(first_date, str) else first_date
    second_date_obj = datetime.datetime.strptime(second_date, "%Y-%m-%d") \
        if isinstance(second_date, str) else second_date
    diff = second_date_obj - first_date_obj
    if diff.days < 0:
        return -get_num_of_intervals_between_two_dates(second_date, first_date, date_interval)

    if date_interval == 'day':
        return diff.days
    elif date_interval == 'week':
        return diff.days / 7
    elif date_interval == 'month':
        return (second_date_obj.year - first_date_obj.year) * 12 + \
            second_date_obj.month - first_date_obj.month
    else:
        raise ValueError("Date interval not well defined, only support day, week and month.")


def calculate_date_plus_months(date, num_months):
    """
    Given a date and the number of months, We calculate a new date = date + delta * month.
    Parameters
    ----------
    date: datetime
        A date time object, starting date
    num_months: int
        The number of months

    Returns
    -------
        datetime
        The new date

    """
    m, y = (date.month + num_months) % 12, date.year + (date.month + num_months - 1) // 12
    if not m:
        m = 12
    d = min(date.day, [31,
                       29 if y % 4 == 0 and not y % 400 == 0 else 28, 31, 30, 31, 30, 31, 31, 30,
                       31, 30, 31][m - 1])
    return date.replace(day=d, month=m, year=y)


def calculate_date_plus_date_intervals(start_date, num_date_intervals, date_interval):
    """
    Given a number, a start date and a date interval (day/week/month), we calculate a new date =
    start_date + num_date_intervals * date_interval
    Parameters
    ----------
    num_date_intervals: int
        The number of date interval
    start_date: str or datetime object
        The starting date, format is 2017-02-03
    date_interval: str
        day/week/month

    Returns
    -------
        datetime object
        The calculated new date.

    """
    start_date_obj = convert_str_to_date(start_date)

    if date_interval == 'day':
        return start_date_obj + datetime.timedelta(days=num_date_intervals)
    elif date_interval == 'week':
        return start_date_obj + datetime.timedelta(days=num_date_intervals * 7)
    elif date_interval == 'month':
        return calculate_date_plus_months(start_date_obj, num_date_intervals)
    else:
        raise ValueError("Date interval not well defined, only support day, week and month.")


def convert_str_to_date(date_str):
    """
    Given a date like "2019-05-01", return the datetime object of it.

    Parameters
    ----------
    date_str: str or datetime object
        With format like "2019-05-01"

    Returns
    -------
        datetime object
    """
    return datetime.datetime.strptime(date_str, "%Y-%m-%d") \
        if isinstance(date_str, str) else date_str


def get_last_monday(date_str):
    """
    Given a date like "2019-05-01", which is a Wednesday, return the last Monday before,
    which is 2019-04-29.
    If the date is a Monday, like "2019-04-01", we then return 2019-04-01.
    Parameters
    ----------
    date_str: str
        With format YYYY-MM-DD

    Returns:
    -------
        datetime object
        The last Monday before the given date. If the date is a Monday, then return itself.
    """
    date_obj = convert_str_to_date(date_str)
    return date_obj - datetime.timedelta(days=date_obj.weekday())


def get_parent_path(current_file_path):
    """

    Parameters
    ----------
    current_file_path: str
        The given file path, should be an absolute path

    Returns: str
        The parent path of give file path
    -------

    """

    return os.path.abspath(os.path.join(current_file_path, os.pardir))


def merge_two_dictionaries(dict1, dict2):
    """

    Parameters
    ----------
    dict1: dict
        The first dictionary
    dict2: dict
        The second dictionary

    Returns: dict
        The addition of two dictionaries
    -------

    """
    new_dict = dict1.copy()
    new_dict.update(dict2)
    return new_dict


def is_empty_dataframe(df):
    """
    A simple function to tell whether the passed in df is an empty dataframe or not.
    Parameters
    ----------
    df: pd.DataFrame
        given input dataframe

    Returns
    -------
        boolean
        True if df is none, or if df is an empty dataframe; False otherwise.
    """
    return df is None or (isinstance(df, pd.DataFrame) and df.empty)


def make_performance_table(eval_res_list, keys_to_exclude=None):
    """
    Convert a list of evaluation results (dict) to a DataFrame.

    Parameters
    ----------
    eval_res_list: list[dict]
        A list of dictionaries, each of which will correspond to one row in the final DataFrame.
    keys_to_exclude: list[str]
        A list of columns that should not appear in the final DataFrame.

    Returns
    -------
    df: DataFrame
    """

    keys_to_exclude = set(keys_to_exclude) if keys_to_exclude is not None else set()

    # First collect all possible column names
    all_columns = set()
    for res in eval_res_list:
        for k in res:
            if k in keys_to_exclude:
                continue
            all_columns.add(k)

    # Prepare DataFrame data
    column_to_values = defaultdict(list)
    for res in eval_res_list:
        for col in all_columns:
            column_to_values[col].append(res.get(col))

    df = pd.DataFrame(column_to_values)

    return df


def plot_cross_validation_predictions(
        model_key,
        component_to_cross_validation_results,
        save_to_path=None,
):
    """
    Plot actuals vs. predictions for each fold in cross validation, and for each component.

    Parameters
    ----------
    model_key: tuple
        model_key is a tuple of strs, which jointly uniquely identifies a model.
    component_to_cross_validation_results: dict
        A dictionary that maps a str to a list of dicts.
    save_to_path: str
        If it is not None, then plots will be saved to the given path.

    """

    def plot_for_one_component(ax, component_name, list_of_actuals, list_of_preds):
        """
        Helper function to plot actuals vs. predictions
        for each fold in cross validation, for on single component.

        Parameters
        ----------
        ax: pyplot axis object
        component_name: str
        list_of_actuals: list
            Each element of the list is a 1d numpy array.
        list_of_preds: list
            Each element of the list is a 1d numpy array.
        """
        ax.set_title(component_name)
        start_index = 0

        concat_actuals = []
        for i_fold, preds in enumerate(list_of_preds):
            stop_index = start_index + preds.shape[0]
            t = np.arange(start_index, stop_index)
            ax.plot(t, preds, label='pred-fold=' + str(i_fold))
            start_index = stop_index
            concat_actuals.append(list_of_actuals[i_fold])

        concat_actuals = np.concatenate(concat_actuals)
        ax.plot(concat_actuals, label='actual')
        ax.legend(loc='best')

    file_name_base = '_'.join(model_key)

    fig, axes_list = plt.subplots(
        nrows=len(component_to_cross_validation_results), ncols=1, figsize=(
            16, 8,
        ),
    )

    fig.suptitle(file_name_base, fontsize=16)

    for i, component_name in enumerate(component_to_cross_validation_results.keys()):
        list_of_actuals = [
            res['y_true']
            for res in component_to_cross_validation_results[component_name]
        ]
        list_of_preds = [
            res['y_pred']
            for res in component_to_cross_validation_results[component_name]
        ]
        plot_for_one_component(axes_list[i], component_name, list_of_actuals, list_of_preds)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_to_path is not None:
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)
        file_path = os.path.join(save_to_path, file_name_base + '.png')
        fig.savefig(file_path, format='png', dpi=300)

    plt.show()


def plot_actuals_vs_preds(model_key, actuals, preds, pred_start_index=0, save_to_path=None):
    """
    Plot predictions on top of actual data.

    Parameters
    ----------
    model_key: tuple
        model_key is a tuple of strs, which jointly uniquely identifies a model.
    actuals: 1d numpy array
    preds: 1d numpy array
        Note that the length of preds does not need to match that of actuals.
    pred_start_index: int
        The starting index (relative to actuals) to plot preds.
    save_to_path: str
        If it is not None, save plots to the path.

    """

    file_name_base = '_'.join(model_key)

    plt.figure(figsize=(16, 5))

    plt.plot(actuals, label='historical')
    plt.plot(range(pred_start_index, pred_start_index + len(preds)), preds, label='prediction')
    plt.legend(loc='best')

    plt.title(file_name_base, fontsize=16)

    plt.tight_layout()

    if save_to_path is not None:
        if not os.path.isdir(save_to_path):
            os.mkdir(save_to_path)
        file_path = os.path.join(save_to_path, file_name_base + '.png')
        plt.savefig(file_path, format='png', dpi=300)

    plt.show()


def shift_by_days(datestr, num_days):
    """
    Shift a date by numder of days.

    Parameters
    ----------
    datestr: str
        datestr should follow the format 'YYYY-MM-DD'
    num_days: int

    Returns
    -------
    new_datestr: str
        new_datestr should follow the format 'YYYY-MM-DD'
    """
    date_obj = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    new_date_obj = date_obj + datetime.timedelta(days=num_days)
    new_datestr = str(new_date_obj.date())
    return new_datestr


def check_dates_exist(df, datestrs):
    """
    Check if all dates exist in the given DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame's index should be date strings that follow 'YYYY-MM-DD'
    datestrs: list[str]
        Each str in the list should follow 'YYYY-MM-DD'

    Return
    ------
    ret: boolean
        if True, it means all date strings appear in the DataFrame;
        False if there is any missing date string.
    """

    ret = True

    all_indexes = set(df.index.values.tolist())

    for datestr in datestrs:
        if datestr not in all_indexes:
            ret = False
            break

    return ret

# class SuppressStdOutStdErr(object):
#     """
#     A context manager for doing a "deep suppression" of stdout and stderr in
#     Python, i.e. will suppress all print, even if the print originates in a
#     compiled C/Fortran sub-function.
#        This will not suppress raised exceptions, since exceptions are printed
#     to stderr just before a script exits, and after the context manager has
#     exited (at least, I think that is why it lets exceptions through).
#     """
#
#     def __init__(self):
#         # Open a pair of null files
#         self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
#         # Save the actual stdout (1) and stderr (2) file descriptors.
#         self.save_fds = [os.dup(1), os.dup(2)]
#
#     def __enter__(self):
#         # Assign the null pointers to stdout and stderr.
#         os.dup2(self.null_fds[0], 1)
#         os.dup2(self.null_fds[1], 2)
#
#     def __exit__(self, *_):
#         # Re-assign the real stdout/stderr back to (1) and (2)
#         os.dup2(self.save_fds[0], 1)
#         os.dup2(self.save_fds[1], 2)
#         # Close the null files
#         for fd in self.null_fds + self.save_fds:
#             os.close(fd)
