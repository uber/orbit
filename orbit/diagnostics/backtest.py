import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import inspect
import tqdm

from .metrics import smape, wmape, mape, mse, mae, rmsse
from ..exceptions import BacktestException
from ..constants.constants import TimeSeriesSplitSchemeNames
from ..constants.palette import QualitativePalette

from itertools import product
from collections.abc import Mapping, Iterable
from orbit.diagnostics.metrics import smape, mape, wmape


class TimeSeriesSplitter(object):
    """Cross validation splitter for time series data"""

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
        self._test_predicted = []
        self._train_actual = []
        self._train_predicted = []

        # init df for actuals and predictions
        self._predicted_df = pd.DataFrame(
            {}, columns=['date', 'split_key', 'training_data', 'actuals', 'prediction']
        )

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
        return self._splitter

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
        for train_df, test_df, scheme, key in splitter.split():
            model_copy = deepcopy(model)
            model_copy.fit(train_df)
            train_predictions = model_copy.predict(train_df)
            test_predictions = model_copy.predict(test_df)

            # set attributes
            self._fitted_models.append(model_copy)
            self._splitter_scheme.append(scheme)
            self._test_actual = np.concatenate((self._test_actual, test_df[response_col].to_numpy()))
            self._test_predicted = np.concatenate((self._test_predicted, test_predictions['prediction'].to_numpy()))
            self._train_actual = np.concatenate((self._train_actual, train_df[response_col].to_numpy()))
            self._train_predicted = np.concatenate((self._train_predicted, train_predictions['prediction'].to_numpy()))

            # set df attribute
            # join train
            train_dates = train_df[date_col].rename('date', axis='columns')
            train_response = train_df[response_col].rename('actuals', axis='columns')
            train_values = pd.concat((train_dates, train_response, train_predictions['prediction']), axis=1)
            train_values['training_data'] = True
            # join test
            test_dates = test_df[date_col].rename('date', axis='columns')
            test_response = test_df[response_col].rename('actuals', axis='columns')
            test_values = pd.concat((test_dates, test_response, test_predictions['prediction']), axis=1)
            test_values['training_data'] = False
            # union train/test
            both_values = pd.concat((train_values, test_values), axis=0)
            both_values['split_key'] = key
            # union each splits
            self._predicted_df = pd.concat((self._predicted_df, both_values), axis=0).reset_index(drop=True)

    def get_predicted_df(self):
        return self._predicted_df

    def get_fitted_models(self):
        return self._fitted_models

    def get_scheme(self):
        return self._splitter_scheme

    @staticmethod
    def _get_metric_callable_signature(metric_callable):
        metric_args = inspect.getfullargspec(metric_callable)
        args = metric_args.args
        return set(args)

    def _validate_metric_callables(self, metrics):
        for metric in metrics:
            metric_signature = self._get_metric_callable_signature(metric)
            if metric_signature == {'actual', 'predicted'}:
                continue
            elif metric_signature.issubset({'test_actual', 'test_predicted', 'train_actual', 'train_predicted'}):
                continue
            else:
                raise BacktestException("metric callable does not have a supported function signature")

    def _evaluate_test_metric(self, metric):
        # signature already validated in `self._validate_metric_callable()` so the following
        # values for metric_signature already are only for valid signatures
        metric_signature = self._get_metric_callable_signature(metric)
        if metric_signature == {'actual', 'predicted'}:
            eval_out = metric(actual=self._test_actual, predicted=self._test_predicted)
        else:
            # get signature
            _valid_args = ['_' + x for x in metric_signature]  # add leading underscore to found signatures
            valid_arg_vals = [getattr(self, x) for x in _valid_args]  # get private variable eg `self._test_actual`
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

        self._validate_metric_callables(metrics)

        # test data metrics
        eval_out_list = list()

        for metric in metrics:
            eval_out = self._evaluate_test_metric(metric)
            eval_out_list.append(eval_out)

        metrics_str = [x.__name__ for x in metrics]  # metric names string
        self._score_df = pd.DataFrame(metrics_str, columns=['metric_name'])
        self._score_df['metric_values'] = eval_out_list
        self._score_df['is_training_metric'] = False

        # for metric evaluation with combined train and test
        if include_training_metrics:
            # only supports simple metrics function signature
            metrics = list(filter(lambda x: self._get_metric_callable_signature(x) == {'actual', 'predicted'}, metrics))
            train_eval_out_list = list()
            for metric in metrics:
                eval_out = metric(actual=self._train_actual, predicted=self._train_predicted)
                train_eval_out_list.append(eval_out)

            metrics_str = [x.__name__ for x in metrics]  # metric names string
            train_score_df = pd.DataFrame(metrics_str, columns=['metric_name'])
            train_score_df['metric_values'] = train_eval_out_list
            train_score_df['is_training_metric'] = True

            self._score_df = pd.concat((self._score_df, train_score_df), axis=0).reset_index(drop=True)

        return self._score_df


def grid_search_orbit(param_grid, model, df, min_train_len,
                      incremental_len, forecast_len, n_splits=None,
                      metrics=None, criteria=None, verbose=True, **kwargs):
    """A gird search unitlity to tune the hyperparameters for orbit models using the orbit.diagnostics.backtest modules.
    Parameters
    ----------
    param_gird : dict
        a dict with candidate values for hyper-params to be tuned
    model : object
        model object
    df : pd.DataFrame
    min_train_len : int
        scheduling parameter in backtest
    incremental_len : int
        scheduling parameter in backtest
    forecast_len : int
        scheduling parameter in backtest
    n_splits : int
        scheduling parameter in backtest
    metrics : function
        metric function, defaul smape defined in orbit.diagnostics.metrics
    criteria : str
        "min" or "max"; defatul is None ("min")
    verbose : bool

    Return
    ------
    dict:
        best hyperparams
    pd.DataFrame:
        data frame of tuning results

    """
    def _get_params(model):
        # get all the model params for orbit typed models
        params = {}
        for key, val in model.__dict__.items():
            if not key.startswith('_') and key != 'estimator':
                params[key] = val

        for key, val in model.__dict__['estimator'].__dict__.items():
            if not key.startswith('_') and key != 'stan_init':
                params[key] = val

        return params.copy()

    def _yield_param_grid(param_grid):
        # an internal function to mimic the ParameterGrid from scikit-learn
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                                'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        for p in param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    param_list_dict = list(_yield_param_grid(param_grid))
    params = _get_params(model)
    res = pd.DataFrame(param_list_dict)
    metric_values = list()

    for tuned_param_dict in tqdm.tqdm(param_list_dict):
        if verbose:
            print("tuning hyper-params {}".format(tuned_param_dict))

        params_ = params.copy()
        for key, val in tuned_param_dict.items():
            if key not in params_.keys():
                raise Exception("tuned hyper-param {} is not in the model's parameters".format(key))
            else:
                params_[key] = val

        # it is safer to reinstantiate a model object than using deepcopy...
        model_ = model.__class__(**params_)
        bt = BackTester(
            model=model_,
            df=df,
            min_train_len=min_train_len,
            n_splits=n_splits,
            incremental_len=incremental_len,
            forecast_len=forecast_len,
            **kwargs
        )
        bt.fit_predict()
        # TODO: should we assert len(metrics) == 1?
        if metrics is None:
            metrics = smape
        metric_val = bt.score(metrics=[metrics]).metric_values[0]
        if verbose:
            print("tuning metric:{:-.5g}".format(metric_val))
        metric_values.append(metric_val)
    res['metrics'] = metric_values
    if criteria is None:
        criteria = 'min'
    best_params = res[res['metrics'] == res['metrics'].apply(criteria)].drop('metrics', axis=1).to_dict('records')

    return best_params, res
