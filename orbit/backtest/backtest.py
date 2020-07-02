import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from orbit.backtest import metrics as metlib
from orbit.exceptions import BacktestException
from orbit.constants.backtest import BacktestNames
from orbit.constants.palette import QualitativePalette
from orbit.backtest.outcome import to_df


class TimeSeriesSplitter(object):
    """ Split time series observations into train-test style
    """
    # FIXME: enforce date_col as an input?
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
        date_col : str
            optional for user to provide date columns; note that it stills uses discrete index
            as splitting scheme while `date_col` is used for better visualization only

        Attributes
        ----------
        _schemes : list of dict
            each element describes meta data to split train and test set
        """

        self.df = df.copy()
        self.min_train_len = min_train_len
        self.incremental_len = incremental_len
        self.forecast_len = forecast_len
        self.n_splits = n_splits
        self.window_type = window_type
        self.date_col = date_col

        # defaults
        self._set_defaults()

        # validate
        self._validate_params()

        # init meta data of how to split
        self._schemes = []

        # timeseries cross validation split
        self._set_schemes()

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

    def _set_schemes(self):
        test_end_min = self.min_train_len - 1
        test_end_max = self._df_length - self.forecast_len
        test_seq = range(test_end_min, test_end_max, self.incremental_len)

        schemes = []
        for train_end_idx in test_seq:
            new_scheme = {}
            train_start_idx = train_end_idx - self.min_train_len + 1 \
                if self.window_type == 'rolling' else 0
            new_scheme[BacktestNames.TRAIN_IDX.value] = range(
                train_start_idx, train_end_idx + 1)
            new_scheme[BacktestNames.TEST_IDX.value] = range(
                train_end_idx + 1, train_end_idx + self.forecast_len + 1)

            schemes.append(new_scheme)

        self._schemes = schemes
        # enforce n_splits to match scheme in case scheme is determined by min_train_len
        self.n_splits = len(schemes)

    def get_schemes(self):
        return self._schemes

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
            derived from self._scheme
        split_key : int
             index of the iteration
        """
        for sch in self._schemes:
            train_df = self.df.iloc[sch[BacktestNames.TRAIN_IDX.value], :] \
                .reset_index(drop=True)
            test_df = self.df.iloc[sch[BacktestNames.TEST_IDX.value], :] \
                .reset_index(drop=True)

            yield train_df, test_df, sch

    def __str__(self):
        message = ""
        for idx, sch in enumerate(self._schemes):
            # print train/test start/end indices
            tr_start = list(sch[BacktestNames.TRAIN_IDX.value])[0]
            tr_end = list(sch[BacktestNames.TRAIN_IDX.value])[-1]
            tt_start = list(sch[BacktestNames.TEST_IDX.value])[0]
            tt_end = list(sch[BacktestNames.TEST_IDX.value])[-1]
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

    def plot(self, lw=20, figsize=None):
        if figsize is None:
            figsize = (20, self.n_splits)
        _, ax = plt.subplots(figsize=figsize)
        # visualize the train/test windows for each split
        for idx, sch in enumerate(self._schemes):
            # fill in indices with the training/test groups
            tr_indices = list(sch[BacktestNames.TRAIN_IDX.value])
            tt_indices = list(sch[BacktestNames.TEST_IDX.value])

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


class Backtest(object):
    '''Object used to backtest
    Parameters
    ----------
    splitter: orbit.backtest.bacltest.TimeSeriesSplitter
        splitter object which describes the scheme to split train and test set
    '''

    def __init__(self, splitter):
        """Initializes Backtest object with DataFrame and splits data

        Parameters
        ----------
        splitter : instance of TimeSeriesSplitter
        """
        # init predictions df
        # grouped by model, split, time
        self._predicted_df = pd.DataFrame()
        self.splitter = deepcopy(splitter)

    def fit_score(self, model, response_col, predicted_col='prediction', metrics=None, insample_predict=False,
                  save_model=False, model_callback=None, fit_callback=None,
                  predict_callback=None, fit_args=None, predict_args=None):
        """
        Parameters
        ----------
        model : object
            arbitrary instantiated model object
        response_col: str
            column label of data frame define the response
        predicted_col : str
            column indicates the prediction label
        metrics : dict
            dictionary of callables f which score performance by f(outcome); outcome is a dictionary
        insample_predict : bool
            logic whether conduct the insample prediction on training data
        save_model : bool
            logic whether save the fitted models for each backtest iteration
        model_callback : callable
            optional callback to adapt model object to work with `orbit.backtest`. Particularly,
            the object needs a `fit()` and `predict()` methods if not exists.
        fit_callback : callable
            optional callback to adapt data to work with `model`'s `fit()` method
        predict_callback : callable
            optional callback to adapt prediction results to work with `orbit.backtest`
        fit_args :
            additional kwargs to be passed to the `fit_callback` function
        predict_args :
            additional kwargs to be passed to the `predict_callback` function
        """

        # TODO: Consider fit response_col from model but set score takes another separately
        if fit_args is None:
            fit_args = {}

        self.insample_predict = insample_predict
        # FIXME: also assert model.date_col == self.splitter.date_col if exists
        self.date_col = self.splitter.date_col if self.splitter.date_col else None
        self.predicted_col = predicted_col
        self.response_col = response_col

        self._fit(
            model=model,
            save_model=save_model,
            model_callback=model_callback,
            fit_callback=fit_callback,
            predict_callback=predict_callback,
            fit_args=fit_args,
            predict_args=predict_args
        )

        self._set_score(metrics=metrics)

    def _fit(self, model, save_model=False,
             model_callback=None, fit_callback=None, predict_callback=None,
             fit_args=None, predict_args=None):
        """Fits the splitted data to the model and predicts

        Parameters
        ----------
        model : object
            arbitrary instantiated model object
        save_model : bool
            logic whether save the fitted models for each backtest iteration
        model_callback : callable
            optional callback to adapt model object to work with `orbit.backtest`. Particularly,
            the object needs a `fit()` and `predict()` methods if not exists.
        fit_callback : callable
            optional callback to adapt data to work with `model`'s `fit()` method
        predict_callback : callable
            optional callback to adapt prediction results to work with `orbit.backtest`
        fit_args :
            additional kwargs to be passed to `fit_callback`
        predict_args :
            additional kwargs to be passed to `predict_callback`

        Returns
        -------
        pd.DataFrame
            DataFrame with Predictions

        """


        # todo: kwargs need to be parsed to know
        #   which callbacks they belong to
        #   alternatively, dict for each callback so we know clearly

        bt_models = []
        pred_outcomes = []

        for train_df, test_df, scheme in self.splitter.split():
            # if we need to extend model to work with backtest
            if model_callback is not None:
                model = model_callback(model)

            # fit model
            if fit_callback is None:
                model.fit(train_df)
            else:
                fit_callback(model, train_df, **fit_args)

            if predict_callback is None:
                predicted_df = model.predict(test_df)
            else:
                predicted_df = predict_callback(model, test_df, **predict_args)

            # packing result into a outcome dict
            outcome = {
                "train_actual": train_df[self.response_col].values,
                "train_pred": None,
                "train_dts": None,
                "test_pred": predicted_df[self.predicted_col].values,
                "test_actual": test_df[self.response_col].values,
                "test_dts": None,
                "train_end_dt": None,
                "n_train": train_df.shape[0],
                "n_test": test_df.shape[0],
            }

            if self.insample_predict:
                if predict_callback is None:
                    predicted_df = model.predict(train_df)
                else:
                    predicted_df = predict_callback(model, train_df, **predict_args)
                outcome['train_pred'] = predicted_df[self.predicted_col]

            if self.date_col:
                outcome["train_end_dt"] = train_df[self.date_col].values[-1]
                outcome["train_dts"] = train_df[self.date_col].values
                outcome["test_dts"] = test_df[self.date_col].values

            if save_model:
                bt_models.append(deepcopy(model))
            pred_outcomes.append(outcome)

        self._pred_outcomes = pred_outcomes
        self._bt_models = bt_models

        return pred_outcomes

    def _set_score(self, metrics=None):
        """
        Parameters
        ----------
        metrics : dict
            dictionary of callables which score performance by (actual, predicted)

        Returns
        -------
        pd.DataFrame
            DataFrame with scores computed with metrics function provided

        """
        # TODO: recover how to evaluate metrics with different steps
        score_df = self._score_by(metrics=metrics)
        self._score_df = score_df
        return score_df

    def _score_by(self, metrics=None):
        """

        Parameters
        ----------
        response_col : str
            column label of data frame define the response of the model
        predicted_col : str
            optional callback to adapt data to work with `model`'s `fit()` method
        metrics : dict
            dictionary of callables which score performance by (actual, predicted)

        Returns
        -------
        pd.DataFrame
            DataFrame with scores computed with metrics function provided

        """
        # TODO: recover metrics calculated with group by

        if metrics is None:
            metrics = ['wmape', 'smape', 'rmsse']

        # pred_df_grouped = self._predicted_df.groupby(by=groupby)
        score_df = pd.DataFrame({
            'n_splits':  len(self._pred_outcomes),
            'forecast_len':  self.splitter.forecast_len,
            'incremental_len':  self.splitter.incremental_len,
        }, index=[0])

        # multiple models or step segmentation
        # if groupby is not None:
        for metric_name in metrics:
            metric_method = getattr(metlib, metric_name)
            z = np.fromiter(map(metric_method, self._pred_outcomes), dtype=np.float32)
            score_df[metric_name] = np.mean(z)

        return score_df

    def _append_split_meta(self):
        pass

    def _append_model_meta(self):
        pass

    def get_predictions(self, insample=False):
        result = []
        for idx, poc in enumerate(self._pred_outcomes):
            df = to_df(poc, self.date_col, self.response_col)
            if not insample:
                df = df[df[BacktestNames.TRAIN_TEST_PARTITION.value] == 'test'].reset_index(drop=True)
            df['split_key'] = idx
            result.append(df)
        result = pd.concat(result, axis=0)
        return result

        # TODO: implement include_split_meta
        return self._predicted_df[self._predicted_df[BacktestNames.TRAIN_TEST_PARTITION.value] == 'test'].\
            drop(columns=[BacktestNames.TRAIN_TEST_PARTITION.value]).reset_index(drop=True)

    # def get_insample_predictions(self, include_split_meta=False):
    #     # TODO: implement include_split_meta
    #     if not self.insample_predict:
    #         raise BacktestException('insample_predict has to be True to obtain insample predictions...')
    #
    #     return self._predicted_df[self._predicted_df[BacktestNames.TRAIN_TEST_PARTITION.value] == 'train'].\
    #         drop(columns=[BacktestNames.TRAIN_TEST_PARTITION.value]).reset_index(drop=True)

    def get_scores(self, include_model_meta=False):
        # TODO: implement include_split_meta
        return self._score_df

    # def get_insample_scores(self, include_model_meta=False):
    #     # TODO: implement include_split_meta
    #     if not self.insample_predict:
    #         raise BacktestException('insample_predict has to be True to obtain insample scores...')
    #     return self._score_df[self._score_df[BacktestNames.TRAIN_TEST_PARTITION.value] == 'train'].\
    #         drop(columns=[BacktestNames.TRAIN_TEST_PARTITION.value]).reset_index(drop=True)

    def get_fitted_models(self):
        return self._bt_models
