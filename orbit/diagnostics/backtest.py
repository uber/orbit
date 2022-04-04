import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import inspect

from .metrics import smape, wmape, mape, mse, mae, rmsse
from ..exceptions import BacktestException
from ..constants.constants import TimeSeriesSplitSchemeKeys, BacktestFitKeys
from ..constants.palette import OrbitPalette as OrbitPal
from orbit.utils.plot import orbit_style_decorator


class TimeSeriesSplitter(object):
    """Cross validation splitter for time series data"""

    def __init__(
        self,
        df,
        forecast_len=1,
        incremental_len=None,
        n_splits=None,
        min_train_len=None,
        window_type="expanding",
        date_col=None,
    ):
        """Initializes object with DataFrame and splits data

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame object containing time index, response, and other features
        forecast_len : int
            forecast length; default as 1
        incremental_len : int
            the number of observations between each successive backtest period; default as forecast_len
        n_splits : int; default None
            number of splits; when n_splits is specified, min_train_len will be ignored
        min_train_len : int
            the minimum number of observations required for the training period
        window_type : {'expanding', 'rolling }; default 'expanding'
            split scheme
        date_col : str
            optional for user to provide date columns; note that it stills uses discrete index
            as splitting scheme while `date_col` is used for better visualization only

        Attributes
        ----------
        _split_scheme : dict{split_meta}
            meta data of ways to split train and test set
        """

        self.df = df.copy()
        self.min_train_len = min_train_len
        self.incremental_len = incremental_len
        self.forecast_len = forecast_len
        self.n_splits = n_splits
        self.window_type = window_type
        self.date_col = None
        self.dt_array = None

        if date_col is not None:
            self.date_col = date_col
            # support cases for multiple observations
            self.dt_array = pd.to_datetime(np.sort(self.df[self.date_col].unique()))

        self._set_defaults()

        # validate
        self._validate_params()

        # init meta data of how to split
        self._split_scheme = {}

        # timeseries cross validation split
        self._set_split_scheme()

    def _set_defaults(self):
        if self.date_col is None:
            self._full_len = self.df.shape[0]
        else:
            self._full_len = len(self.dt_array)

        if self.incremental_len is None:
            self.incremental_len = self.forecast_len

        # if n_splits is specified, set min_train_len internally
        if self.n_splits:
            # if self.n_splits == 1:
            #     # set incremental_len internally if it's None
            #     # this is just to dodge error and it's not used actually
            self.min_train_len = (
                self._full_len
                - self.forecast_len
                - (self.n_splits - 1) * self.incremental_len
            )

    def _validate_params(self):
        if self.min_train_len is None and self.n_splits is None:
            raise BacktestException("min_train_len and n_splits cannot both be None...")

        if self.window_type not in [
            TimeSeriesSplitSchemeKeys.SPLIT_TYPE_EXPANDING.value,
            TimeSeriesSplitSchemeKeys.SPLIT_TYPE_ROLLING.value,
        ]:
            raise BacktestException("unknown window type...")

        # forecast length invalid
        if self.forecast_len <= 0:
            raise BacktestException("holdout period length must be positive...")

        # train + test length cannot be longer than df length
        if self.min_train_len + self.forecast_len > self._full_len:
            raise BacktestException(
                "required time span is more than the full data frame..."
            )

        if self.n_splits is not None and self.n_splits < 1:
            raise BacktestException("n_split must be a positive number")

        if self.date_col:
            if self.date_col not in self.df.columns:
                raise BacktestException("date_col not found in df provided.")

    def _set_split_scheme(self):
        """set meta data of ways to split train and test set"""
        test_end_min = self.min_train_len - 1
        test_end_max = self._full_len - self.forecast_len
        test_seq = range(test_end_min, test_end_max, self.incremental_len)

        split_scheme = {}
        # note that
        # in range representation, inclusive bound on the left and exclusive bound on the right is used
        # in date periods representation, both bound are inclusive to work around limitation on df[date_col][idx]
        for i, train_end_idx in enumerate(test_seq):
            split_scheme[i] = {}
            train_start_idx = (
                train_end_idx - self.min_train_len + 1
                if self.window_type
                == TimeSeriesSplitSchemeKeys.SPLIT_TYPE_ROLLING.value
                else 0
            )
            split_scheme[i][TimeSeriesSplitSchemeKeys.TRAIN_IDX.value] = range(
                train_start_idx, train_end_idx + 1
            )
            split_scheme[i][TimeSeriesSplitSchemeKeys.TEST_IDX.value] = range(
                train_end_idx + 1, train_end_idx + self.forecast_len + 1
            )

            if self.date_col is not None:
                split_scheme[i]["train_period"] = (
                    self.dt_array[train_start_idx],
                    self.dt_array[train_end_idx],
                )
                split_scheme[i]["test_period"] = (
                    self.dt_array[train_end_idx + 1],
                    self.dt_array[train_end_idx + self.forecast_len],
                )

        self._split_scheme = split_scheme
        # enforce n_splits to match scheme in case scheme is determined by min_train_len
        self.n_splits = len(split_scheme)

    def get_scheme(self):
        return deepcopy(self._split_scheme)

    def split(self):
        """
        Returns
        -------
        iterables with (train_df, test_df, scheme, split_key) where
        train_df : pd.DataFrame
            data split for training
        test_df : pd.DataFrame
            data split for testing/validation
        scheme : dict
            derived from self._split_scheme
        split_key : int
             index of the iteration
        """
        if self.date_col is None:
            for split_key, scheme in self._split_scheme.items():
                train_df = self.df.iloc[
                    scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value], :
                ].reset_index(drop=True)
                test_df = self.df.iloc[
                    scheme[TimeSeriesSplitSchemeKeys.TEST_IDX.value], :
                ].reset_index(drop=True)

                yield train_df, test_df, scheme, split_key
        else:
            for split_key, scheme in self._split_scheme.items():
                train_df = self.df.loc[
                    (self.df[self.date_col] >= scheme["train_period"][0])
                    & (self.df[self.date_col] <= scheme["train_period"][1]),
                    :,
                ].reset_index(drop=True)
                test_df = self.df.loc[
                    (self.df[self.date_col] >= scheme["test_period"][0])
                    & (self.df[self.date_col] <= scheme["test_period"][1]),
                    :,
                ].reset_index(drop=True)

                yield train_df, test_df, scheme, split_key

    # TODO: adapt to date col if provided
    def __str__(self):
        message = ""
        for idx, scheme in self._split_scheme.items():
            # print train/test start/end indices
            tr_start = list(scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value])[0]
            tr_end = list(scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value])[-1]
            tt_start = list(scheme[TimeSeriesSplitSchemeKeys.TEST_IDX.value])[0]
            tt_end = list(scheme[TimeSeriesSplitSchemeKeys.TEST_IDX.value])[-1]
            message += (
                f"\n------------ Fold: ({idx + 1} / {self.n_splits})------------\n"
            )
            if self.date_col is None:
                message += f"Train start index: {tr_start} Train end index: {tr_end}\n"
                message += f"Test start index: {tt_start} Test end index: {tt_end}\n"
            else:
                tr_start_date = scheme["train_period"][0]
                tr_end_date = scheme["train_period"][1]
                tt_start_date = scheme["test_period"][0]
                tt_end_date = scheme["test_period"][1]

                message += (
                    f"Train start date: {tr_start_date} Train end date: {tr_end_date}\n"
                )
                message += (
                    f"Test start date: {tt_start_date} Test end date: {tt_end_date}\n"
                )
        return message

    @orbit_style_decorator
    def plot(self, fig_width=20, show_index=False, strftime_fmt="%Y-%m-%d"):
        """
        Parameters
        ----------
        fig_width : float
        show_index : bool
        strftime_fmt : str

        Returns
        -------
        matplotlib axes object
        """
        _, ax = plt.subplots(figsize=(fig_width, self.n_splits))
        # visualize the train/test windows for each split
        tr_start = list()
        tr_len = list()
        # technically should be just self.forecast_len
        tt_len = list()
        yticks = list(range(self.n_splits))
        for idx, scheme in self._split_scheme.items():
            # fill in indices with the training/test groups
            tr_start.append(list(scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value])[0])
            tr_len.append(len(list(scheme[TimeSeriesSplitSchemeKeys.TRAIN_IDX.value])))
            tt_len.append(self.forecast_len)

        tr_start = np.array(tr_start)
        tr_len = np.array(tr_len)

        # ax.barh(yticks, tr_start, align='center', height=.5, color='black', alpha=0.5)
        ax.barh(
            yticks,
            tr_len,
            align="center",
            height=0.5,
            left=tr_start,
            color=OrbitPal.BLUE.value,
            label="train",
        )
        ax.barh(
            yticks,
            tt_len,
            align="center",
            height=0.5,
            left=tr_start + tr_len,
            color=OrbitPal.ORANGE.value,
            label="test",
        )

        if not show_index and self.date_col is not None:
            xticks_loc = np.array(ax.get_xticks(), dtype=int)
            new_xticks_loc = np.linspace(
                0, len(self.dt_array) - 1, num=len(xticks_loc)
            ).astype(int)
            dt_xticks = self.dt_array[new_xticks_loc]
            dt_xticks = dt_xticks.strftime(strftime_fmt)
            ax.set_xticks(new_xticks_loc)
            ax.set_xticklabels(dt_xticks)

        # some formatting parameters
        middle = 15
        large = 20

        ax.set_yticks(yticks)
        ax.set_ylabel("Split #", fontsize=large)
        ax.invert_yaxis()
        # ax.grid(which="both", color='grey', alpha=0.5)
        ax.tick_params(axis="x", which="major", labelsize=middle)
        ax.set_title("Train/Test Split Scheme", fontsize=large)
        return ax


class BackTester(object):
    """Used to iteratively fit model on a given data splitter"""

    _default_metrics = [smape, wmape, mape, mse, mae, rmsse]

    def __init__(self, model, df, **splitter_kwargs):
        self.model = model
        self.df = df

        self._splitter_args = splitter_kwargs

        # create splitter object to split data
        self._splitter = None
        self._set_splitter()

        # init private vars
        self._n_splits = 0
        self._set_n_splits()
        self._test_actual = []
        self._test_prediction = []
        self._train_actual = []
        self._train_prediction = []

        # # init df for actual and predictions
        # self._predicted_df = pd.DataFrame(
        #     {}, columns=[
        #         BacktestFitKeys.DATE.value,
        #         BacktestFitKeys.SPLIT_KEY.value,
        #         BacktestFitKeys.TRAIN_FLAG.value,
        #         BacktestFitKeys.ACTUAL.value,
        #         BacktestFitKeys.PREDICTED.value
        #     ]
        # )

        # score df
        self._score_df = pd.DataFrame()

        self._fitted_models = list()
        self._splitter_scheme = list()

    def _make_splitter(self):
        df = self.df
        date_col = self.model.date_col
        splitter_args = self._splitter_args
        splitter = TimeSeriesSplitter(df=df, date_col=date_col, **splitter_args)
        return splitter

    def _set_splitter(self):
        self._splitter = self._make_splitter()

    def get_splitter(self):
        return deepcopy(self._splitter)

    def _set_n_splits(self):
        split_scheme = self._splitter.get_scheme()
        n_splits = len(split_scheme.keys())
        self._n_splits = n_splits

    def fit_predict(self):
        """Fit and predict on each data split and set predicted_df

        Since this part of the backtesting is generally the most expensive, BackTester
        breaks up fit/predict and scoring into two separate calls

        Returns
        -------
        None

        """
        splitter = self._splitter
        model = self.model
        response_col = model.response_col
        date_col = model.date_col
        output_res = list()
        for train_df, test_df, scheme, key in splitter.split():
            model_copy = deepcopy(model)
            model_copy.fit(train_df)
            train_predictions = model_copy.predict(train_df)
            test_predictions = model_copy.predict(test_df)
            all_pred_cols = [x for x in train_predictions.columns if x != date_col]

            # set attributes
            self._fitted_models.append(model_copy)
            self._splitter_scheme.append(scheme)
            self._test_actual = np.concatenate(
                (self._test_actual, test_df[response_col].to_numpy())
            )
            self._test_prediction = np.concatenate(
                (
                    self._test_prediction,
                    test_predictions[BacktestFitKeys.PREDICTED.value].to_numpy(),
                )
            )
            self._train_actual = np.concatenate(
                (self._train_actual, train_df[response_col].to_numpy())
            )
            self._train_prediction = np.concatenate(
                (
                    self._train_prediction,
                    train_predictions[BacktestFitKeys.PREDICTED.value].to_numpy(),
                )
            )

            # set df attribute
            # join train
            train_dates = train_df[date_col].rename(BacktestFitKeys.DATE.value)
            train_response = train_df[response_col].rename(BacktestFitKeys.ACTUAL.value)
            train_values = pd.concat(
                (train_dates, train_response, train_predictions[all_pred_cols]), axis=1
            )
            train_values[BacktestFitKeys.TRAIN_FLAG.value] = True
            # join test
            test_dates = test_df[date_col].rename(BacktestFitKeys.DATE.value)
            test_response = test_df[response_col].rename(BacktestFitKeys.ACTUAL.value)
            test_values = pd.concat(
                (test_dates, test_response, test_predictions[all_pred_cols]), axis=1
            )
            test_values[BacktestFitKeys.TRAIN_FLAG.value] = False
            # union train/test
            both_values = pd.concat((train_values, test_values), axis=0)
            both_values[BacktestFitKeys.SPLIT_KEY.value] = key
            output_res.append(both_values)
        # union each splits
        self._predicted_df = pd.concat(output_res, axis=0).reset_index(drop=True)
        # # recast to expected dtype
        # self._predicted_df[BacktestFitKeys.TRAIN_FLAG.value] = \
        #     self._predicted_df[BacktestFitKeys.TRAIN_FLAG.value].astype('bool')
        # self._predicted_df[BacktestFitKeys.SPLIT_KEY.value] = \
        #     self._predicted_df[BacktestFitKeys.SPLIT_KEY.value].astype('int16')

    def get_predicted_df(self):
        return self._predicted_df.copy()

    def get_fitted_models(self):
        return deepcopy(self._fitted_models)

    def get_scheme(self):
        return deepcopy(self._splitter_scheme)

    def plot_scheme(self, **kwargs):
        """Plot embedded scheme within the backtester object"""
        self._splitter.plot(**kwargs)

    @staticmethod
    def _get_metric_callable_signature(metric_callable):
        metric_args = inspect.getfullargspec(metric_callable)
        args = metric_args.args
        return set(args)

    def _validate_metric_callables(self, metrics):
        for metric in metrics:
            metric_signature = self._get_metric_callable_signature(metric)
            if metric_signature == {
                BacktestFitKeys.ACTUAL.value,
                BacktestFitKeys.PREDICTED.value,
            }:
                continue
            elif metric_signature.issubset(
                {
                    BacktestFitKeys.TEST_ACTUAL.value,
                    BacktestFitKeys.TEST_PREDICTED.value,
                    BacktestFitKeys.TRAIN_ACTUAL.value,
                    BacktestFitKeys.TRAIN_PREDICTED.value,
                }
            ):
                continue
            else:
                raise BacktestException(
                    "metric callable does not have a supported function signature"
                )

    def _evaluate_test_metric(self, metric):
        # signature already validated in `self._validate_metric_callable()` so the following
        # values for metric_signature already are only for valid signatures
        metric_signature = self._get_metric_callable_signature(metric)
        if metric_signature == {
            BacktestFitKeys.ACTUAL.value,
            BacktestFitKeys.PREDICTED.value,
        }:
            eval_out = metric(
                actual=self._test_actual, prediction=self._test_prediction
            )
        else:
            # get signature and match with the private attributes respectively
            # mainly used for cases we need training data into test metrics
            # such as rmsse etc.
            _valid_args = [
                "_" + x for x in metric_signature
            ]  # add leading underscore to found signatures
            valid_arg_vals = [
                getattr(self, x) for x in _valid_args
            ]  # get private variable eg `self._test_actual`
            # dictionary of metric args and arg value
            valid_kwargs = {k: v for k, v in zip(metric_signature, valid_arg_vals)}
            eval_out = metric(**valid_kwargs)

        return eval_out

    def score(self, metrics=None, include_training_metrics=False):
        """Scores predictions using the provided list of metrics

        The following criteria must be met to be a valid callable

        1. All metric callables are required to have `actual` and `predicted` as its input signature. In the case
        that the metric relies on both data from training and testing, it must have `test_actual` and `test_predicted`
        as its input signature and optionally `train_actual` and/or `train_predicted`.

        2. All callables must return a scalar evaluation value

        Parameters
        ----------
        metrics : list
            list of callables for metric evaluation. If None, default to using all
            built-in BackTester metrics. See `BackTester._default_metrics`
        include_training_metrics : bool
            If True, also perform metrics evaluation on training data. Evaluation will only include
            metric callables with `actual` and `predicted` args and will ignore callables with extended args.
            Default, False.

        """
        if metrics is None:
            metrics = self._default_metrics
        elif not isinstance(metrics, list):
            metrics = [metrics]

        self._validate_metric_callables(metrics)

        # test data metrics
        eval_out_list = list()

        for metric in metrics:
            eval_out = self._evaluate_test_metric(metric)
            eval_out_list.append(eval_out)

        metrics_str = [x.__name__ for x in metrics]  # metric names string
        self._score_df = pd.DataFrame(
            metrics_str, columns=[BacktestFitKeys.METRIC_NAME.value]
        )
        self._score_df[BacktestFitKeys.METRIC_VALUES.value] = eval_out_list
        self._score_df[BacktestFitKeys.TRAIN_METRIC_FLAG.value] = False

        # for metric evaluation with combined train and test
        if include_training_metrics:
            # only supports simple metrics function signature
            metrics = list(
                filter(
                    lambda x: self._get_metric_callable_signature(x)
                    == {BacktestFitKeys.ACTUAL.value, BacktestFitKeys.PREDICTED.value},
                    metrics,
                )
            )
            train_eval_out_list = list()
            for metric in metrics:
                eval_out = metric(
                    actual=self._train_actual, prediction=self._train_prediction
                )
                train_eval_out_list.append(eval_out)

            metrics_str = [x.__name__ for x in metrics]  # metric names string
            train_score_df = pd.DataFrame(
                metrics_str, columns=[BacktestFitKeys.METRIC_NAME.value]
            )
            train_score_df[BacktestFitKeys.METRIC_VALUES.value] = train_eval_out_list
            train_score_df[BacktestFitKeys.TRAIN_METRIC_FLAG.value] = True

            self._score_df = pd.concat(
                (self._score_df, train_score_df), axis=0
            ).reset_index(drop=True)

        return self._score_df.copy()
