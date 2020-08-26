import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# from orbit.utils.metrics import (
#     smape, wmape, mape, mse,
# )
from orbit.exceptions import BacktestException
from orbit.constants.constants import TimeSeriesSplitSchemeNames
from orbit.constants.palette import QualitativePalette


class TimeSeriesSplitter(object):
    """ Split time series observations into train-test style
    """

    def __init__(self, df, min_train_len, incremental_len, forecast_len, n_splits=None,
                 window_type='expanding', date_col=None):
        """Initializes object with DataFrame and splits data

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame object containing time index, response, and other features
        min_train_len : int
            the minimum number of observations required for the training period
        incremental_len : int
            the number of observations between each successive backtest period
        forecast_len : int
            forecast length
        n_splits : int; default None
            number of splits; when n_splits is specified, min_train_len will be ignored
        window_type : {'expanding', 'rolling }; default 'expanding'
            split scheme
        date_col : str
            optional for user to provide date columns; note that it stills uses discrete index
            as splitting scheme while `date_col` is used for better visualization only

        Attributes
        ----------
        _split_scheme : dict
            meta data of ways to split train and test set
        """

        self.df = df.copy()
        self.min_train_len = min_train_len
        self.incremental_len = incremental_len
        self.forecast_len = forecast_len
        self.n_splits = n_splits
        self.window_type = window_type
        self.date_col = date_col

        # defaults
        # todo: clean up defaults so you can't set contradicting args
        self._set_defaults()

        # validate
        self._validate_params()

        # init meta data of how to split
        self._split_scheme = {}

        # timeseries cross validation split
        self._set_split_scheme()

    def _set_defaults(self):
        self._df_length = self.df.shape[0]
        # if n_splits is specified, set min_train_len internally
        if self.n_splits:
            self.min_train_len = \
                self._df_length - self.forecast_len - (self.n_splits - 1) * self.incremental_len

    def _validate_params(self):
        if self.window_type not in ['expanding', 'rolling']:
            raise BacktestException('unknown window type...')

        # forecast length invalid
        if self.forecast_len <= 0:
            raise BacktestException('holdout period length must be positive...')

        # train + test length cannot be longer than df length
        if self.min_train_len + self.forecast_len > self._df_length:
            raise BacktestException('required time span is more than the full data frame...')

        if self.n_splits is not None and self.n_splits < 1:
            raise BacktestException('n_split must be a positive number')

        if self.date_col:
            if not self.date_col in self.df.columns:
                raise BacktestException('date_col not found in df provided.')

    def _set_split_scheme(self):
        test_end_min = self.min_train_len - 1
        test_end_max = self._df_length - self.forecast_len
        test_seq = range(test_end_min, test_end_max, self.incremental_len)

        split_scheme = {}
        for i, train_end_idx in enumerate(test_seq):
            split_scheme[i] = {}
            train_start_idx = train_end_idx - self.min_train_len + 1 \
                if self.window_type == 'rolling' else 0
            split_scheme[i][TimeSeriesSplitSchemeNames.TRAIN_IDX.value] = range(
                train_start_idx, train_end_idx + 1)
            split_scheme[i][TimeSeriesSplitSchemeNames.TEST_IDX.value] = range(
                train_end_idx + 1, train_end_idx + self.forecast_len + 1)

        self._split_scheme = split_scheme
        # enforce n_splits to match scheme in case scheme is determined by min_train_len
        self.n_splits = len(split_scheme)

    def get_scheme(self):
        return self._split_scheme

    def split(self):
        """
        Args
        ----

        Returns
        -------
        iterables with (train_df, test_df, scheme, split_key) where
        train_df : pd.DataFrame
            data splitted for training
        test_df : pd.DataFrame
            data splitted for testing/validation
        scheme : dict
            derived from self._split_scheme
        split_key : int
             index of the iteration
        """
        for split_key, scheme in self._split_scheme.items():
            train_df = self.df.iloc[scheme[TimeSeriesSplitSchemeNames.TRAIN_IDX.value], :] \
                .reset_index(drop=True)
            test_df = self.df.iloc[scheme[TimeSeriesSplitSchemeNames.TEST_IDX.value], :] \
                .reset_index(drop=True)

            yield train_df, test_df, scheme, split_key

    def __str__(self):
        message = ""
        for idx, scheme in self._split_scheme.items():
            # print train/test start/end indices
            tr_start = list(scheme[TimeSeriesSplitSchemeNames.TRAIN_IDX.value])[0]
            tr_end = list(scheme[TimeSeriesSplitSchemeNames.TRAIN_IDX.value])[-1]
            tt_start = list(scheme[TimeSeriesSplitSchemeNames.TEST_IDX.value])[0]
            tt_end = list(scheme[TimeSeriesSplitSchemeNames.TEST_IDX.value])[-1]
            message += f"\n------------ Fold: ({idx + 1} / {self.n_splits})------------\n"
            message += f"Train start index: {tr_start} Train end index: {tr_end}\n"
            message += f"Test start index: {tt_start} Test end index: {tt_end}\n"
            if self.date_col is not None:
                tr_start_date = self.df[self.date_col][tr_start]
                tr_end_date = self.df[self.date_col][tr_end]
                tt_start_date = self.df[self.date_col][tt_start]
                tt_end_date = self.df[self.date_col][tt_end]
                message += f"Train start date: {tr_start_date} Train end date: {tr_end_date}\n"
                message += f"Test start date: {tt_start_date} Test end date: {tt_end_date}\n"
        return message

    def plot(self, lw=20, fig_width=20):
        _, ax = plt.subplots(figsize=(fig_width, self.n_splits))
        # visualize the train/test windows for each split
        for idx, scheme in self._split_scheme.items():
            # fill in indices with the training/test groups
            tr_indices = list(scheme[TimeSeriesSplitSchemeNames.TRAIN_IDX.value])
            tt_indices = list(scheme[TimeSeriesSplitSchemeNames.TEST_IDX.value])

            indices = tr_indices + tt_indices
            tr_color = [(QualitativePalette['Bar5'].value)[2]] * len(tr_indices)
            tt_color = [(QualitativePalette['Bar5'].value)[1]] * len(tt_indices)

            # Visualize the results
            ax.scatter(
                indices,
                [idx + 0.5] * len(indices),
                # s=5.0, # not useful
                c=tr_color + tt_color,
                # TODO: consider 's' square marker and other edgecolors
                marker="_",
                lw=lw,
                vmin=-0.2,
                vmax=1.2,
            )

        # Formatting
        # TODO: do a date_col style if date_col is avaliable
        middle = 15
        large = 20

        ax.set_ylabel("Split #", fontsize=large)
        ax.set_yticks(np.arange(self.n_splits) + 0.5)
        ax.set_yticklabels(list(range(self.n_splits)))
        ax.set_ylim(0 - 0.2, self.n_splits + 0.2)
        ax.invert_yaxis()
        ax.grid(which="both", color='grey', alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=middle)
        ax.set_title("Train/Test Split Scheme", fontsize=large)
        return ax


class Evaluator(object):
    """Evaluator for Orbit Model

    Parameters
    ----------
    model : Orbit
        Fitted Orbit model on which to evaluate
    test_df : pd.DataFrame
        DataFrame with actuals and predictions for test period
    train_df : pd.DataFrame
        DataFrame with actuals and predictions for training period.
        For evaluations that require distinction between train and test period [optional]

    """
    def __init__(self, model, test_df, metric='smape', train_df=None):
        self.model = model
        self.test_df = test_df
        self.metric = metric
        self.train_df = train_df

    def score_by_time(self):
        """Returns a DataFrame with scores for each time steps

        This method is only applicable for metrics in which non-aggregated metrics is meaningful.
        For example, MASE cannot reveal metrics at each time step, only in aggregate.
        """
        pass

    def score(self):
        """Returns a scalar evaluation metric"""
        pass


class Backtester(object):
    """Evaluator for each split

    Parameters
    ----------
    model : Orbit
        An initialized orbit model (not fitted)
    splitter : TimeSeriesSplitter
        A timeseries data splitter object
    metrics : list
        List of metrics on which to return results. If None, default value
        is
        ['smape', 'mape']
    """
    def __init__(self, model, splitter, metrics=None):
        self.model = model
        self.splitter = splitter

        # default metrics if not specified
        if metrics is None:
            self.metrics = ['smape', 'mape']
        else:
            self.metrics = metrics

    def score_by_time(self):
        """Returns scores at each time step for each split for each metric"""
        pass

    def score(self):
        """Returns scalar evaluation metric for each split for each metric"""
        pass
