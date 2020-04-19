import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from orbit.utils.metrics import (
    smape, wmape, mape, mse,
)
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
            tr_color = [(QualitativePalette['Q5'].value)[0]] * len(tr_indices)
            tt_color = [(QualitativePalette['Q5'].value)[1]] * len(tt_indices)

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
    """Object used to backtest..."""

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

    def fit_score(self, model, response_col, predicted_col='prediction', metrics=None,
                  insample_predict=False, include_steps=False, model_callback=None, fit_callback=None,
                  predict_callback=None, fit_args=None, predict_args=None):
        """
       Parameters
        ----------
        model : object
            arbitrary instantiated model object
        response_col : str
            column label of data frame define the response of the model
        predicted_col : str
            optional callback to adapt data to work with `model`'s `fit()` method
        metrics : dict
            dictionary of functions `f` which score performance by f(actual_array, predicted_array)
        insample_predict : bool
            logic whether conduct the insample prediction on training data
        include_steps : bool
            logic whether return metrics grouped by steps
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

        # TODO: fit response_col from model but set score takes another separately
        if fit_args is None:
            fit_args = {}

        self.insample_predict = insample_predict

        self._fit(
            model=model,
            insample_predict = insample_predict,
            model_callback=model_callback,
            fit_callback=fit_callback,
            predict_callback=predict_callback,
            fit_args=fit_args,
            predict_args=predict_args
        )

        self._set_score(response_col=response_col, predicted_col=predicted_col,
                        metrics=metrics, include_steps=include_steps)

    def _fit(self, model, insample_predict=False, model_callback=None, fit_callback=None, predict_callback=None,
             fit_args=None, predict_args=None):
        """Fits the splitted data to the model and predicts

        Parameters
        ----------
        model : object
            arbitrary instantiated model object
        insample_predict : bool
            logic whether conduct the insample prediction on training data
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

        Returns
        -------
        pd.DataFrame
            DataFrame with Predictions

        """


        # todo: kwargs need to be parsed to know
        #   which callbacks they belong to
        #   alternatively, dict for each callback so we know clearly

        predicted_df = pd.DataFrame({})
        bt_models = {}

        for train_df, test_df, scheme, split_key in self.splitter.split():
            n_train = train_df.shape[0]
            n_test = test_df.shape[0]

            # if we need to extend model to work with backtest
            if model_callback is not None:
                model = model_callback(model)

            # fit model
            if fit_callback is None:
                model.fit(train_df)
            else:
                fit_callback(model, train_df, **fit_args)

            # predict with model
            x_df = pd.concat((train_df, test_df), axis=0, ignore_index=True) if insample_predict else test_df
            if predict_callback is None:
                predicted_out = model.predict(x_df)
            else:
                predicted_out = predict_callback(model, x_df, **predict_args)

            # predicted results should have same length as test_df
            # if test_df.shape[0] != predicted_out.shape[0]:
            #     raise BacktestException('Prediction length should be the same as test df')

            results = pd.concat((x_df, predicted_out), axis=1)

            # drop duplicate columns
            results = results.loc[:, ~results.columns.duplicated()]
            if self.splitter.date_col:
                results['train_end'] = train_df[self.splitter.date_col].values[-1]
            results['split_key'] = split_key
            results['df_key'] = ['train'] * n_train + ['test'] * n_test if insample_predict else \
                                 ['test'] * n_test
            idx_ = list(range(n_train)) + list(range(n_test)) if insample_predict else list(range(n_test))
            results.set_index([idx_], inplace=True)
            bt_models[split_key] = deepcopy(model)

            predicted_df = pd.concat((predicted_df, results), axis=0)

        predicted_df = predicted_df.reset_index()
        predicted_df = predicted_df.rename(columns={'index': 'steps'})
        predicted_df['steps'] += 1
        self._predicted_df = predicted_df
        self._bt_models = bt_models

        return self._predicted_df

    def _set_score(self, response_col, predicted_col='prediction', metrics=None,
                   include_steps=False):
        """
        Parameters
        ----------
        response_col : str
            column of the data frames
        predicted_col : str
            optional callback to adapt data to work with `model`'s `fit()` method
        metrics : dict
            dictionary of functions `f` which score performance by f(actual_array, predicted_array)
        include_steps : bool
            logic whether return metrics grouped by steps

        Returns
        -------
        pd.DataFrame
            DataFrame with scores computed with metrics function provided

        """
        # TODO: not sure should we restrict use of groupby
        # groupby is controlled internally with include_steps
        groupby = ['df_key', 'steps'] if include_steps else ['df_key']

        score_df = self._score_by(
            response_col=response_col,
            groupby=groupby,
            predicted_col=predicted_col,
            metrics=metrics
        )

        self._score_df = score_df

    def _score_by(self, response_col, groupby, predicted_col='prediction', metrics=None):
        """

        Parameters
        ----------
        response_col : str
            column label of data frame define the response of the model
        groupby : list
            list of columns to be grouped while computing metrics used by pandas.groupby()
        predicted_col : str
            optional callback to adapt data to work with `model`'s `fit()` method
        metrics : dict
            dictionary of functions `f` which score performance by f(actual_array, predicted_array)

        Returns
        -------
        pd.DataFrame
            DataFrame with scores computed with metrics function provided

        """
        if metrics is None:
            metrics = {'wmape': wmape, 'smape': smape}

        predicted_df = self._predicted_df
        score_df = pd.DataFrame({})

        # multiple models or step segmentation
        # if groupby is not None:
        for metric_name, metric_fun in metrics.items():
            score_df[metric_name] = predicted_df \
                .groupby(by=groupby) \
                .apply(lambda x: metric_fun(x[response_col], x[predicted_col]))
        score_df = score_df.reset_index()
        score_df['n_splits'] = self.splitter.n_splits

        # # aggregate without groups
        # else:
        #     for metric_name, metric_fun in metrics.items():
        #         score_df[metric_name] = pd.Series(
        #             metric_fun(predicted_df[response_col], predicted_df[predicted_col])
        #         )
        #     score_df = score_df.reset_index(drop=True)

        return score_df

    def _append_split_meta(self):
        pass

    def _append_model_meta(self):
        pass

    def get_predictions(self, include_split_meta=False):
        # TODO: implement include_split_meta
        return self._predicted_df[self._predicted_df['df_key'] == 'test'].\
            drop(columns=['df_key']).reset_index(drop=True)

    def get_insample_predictions(self, include_split_meta=False):
        # TODO: implement include_split_meta
        if not self.insample_predict:
            raise BacktestException('insample_predict has to be True to obtain insample predictions...')

        return self._predicted_df[self._predicted_df['df_key'] == 'train'].\
            drop(columns=['df_key']).reset_index(drop=True)

    def get_scores(self,  include_model_meta=False):
        # TODO: implement include_split_meta
        return self._score_df[self._score_df['df_key'] == 'test'].\
            drop(columns=['df_key']).reset_index(drop=True)

    def get_insample_scores(self, include_model_meta=False):
        # TODO: implement include_split_meta
        if not self.insample_predict:
            raise BacktestException('insample_predict has to be True to obtain insample scores...')
        return self._score_df[self._score_df['df_key'] == 'train'].\
            drop(columns=['df_key']).reset_index(drop=True)

    def get_fitted_models(self):
        return self._bt_models

    # def _fit_batch(self, models, model_names=None, model_callbacks=None, fit_callbacks=None,
    #                predict_callbacks=None, fit_args=None):
    #     """Runs `_fit()` on a batch of models
    #
    #     Parameters
    #     ----------
    #     models
    #     model_callbacks
    #     fit_callbacks
    #     predict_callbacks
    #     kwargs
    #
    #     Returns
    #     -------
    #
    #     """
    #     # store model object if we need to retrieve later
    #     self._models = models
    #     self._model_names = model_names
    #
    #     batch_size = len(models)
    #
    #     if model_callbacks is None:
    #         model_callbacks = [None] * batch_size
    #     if fit_callbacks is None:
    #         fit_callbacks = [None] * batch_size
    #     if predict_callbacks is None:
    #         predict_callbacks = [None] * batch_size
    #     if fit_args is None:
    #         fit_args = [None] * batch_size
    #
    #     # todo: validate that all the lengths of args align
    #
    #     predicted_df = self._predicted_df
    #
    #     for idx, model in enumerate(models):
    #         each_predicted_df = self._fit(
    #             model,
    #             model_callback=model_callbacks[idx],
    #             fit_callback=fit_callbacks[idx],
    #             predict_callback=predict_callbacks[idx],
    #             fit_args=fit_args[idx]
    #         )  # note this also sets `self._predicted_df` inside `_fit()`
    #
    #         # to retrieve from `self._models` if needed
    #         each_predicted_df['model_idx'] = idx
    #
    #         # some model types raise errors if not converted to string here
    #         # e.g. models with `len()` over-ridden
    #         if model_names:
    #             each_predicted_df['model'] = model_names[idx]
    #         else:
    #             each_predicted_df['model'] = model.__class__.__name__
    #
    #         predicted_df = pd.concat((predicted_df, each_predicted_df), axis=0, ignore_index=True)
    #
    #     self._predicted_df = predicted_df.reset_index(drop=True)
    #
    # def _set_score_batch(self, response_col, predicted_col='prediction',
    #                      metrics=None, include_steps=True):
    #
    #     # groupby determined by `include_steps` bool
    #     groupby = ['model_idx', 'model', 'steps'] \
    #         if include_steps \
    #         else ['model_idx', 'model']
    #
    #     score_df = self._score_by(
    #         response_col=response_col,
    #         groupby=groupby,
    #         predicted_col=predicted_col,
    #         metrics=metrics
    #     )
    #
    #     self._score_df = score_df

    # def fit_score_batch(self, models, response_col, predicted_col='prediction', metrics=None,
    #                     model_names=None, include_steps=False, model_callbacks=None,
    #                     fit_callbacks=None, predict_callbacks=None, fit_args=None):
    #
    #     self._fit_batch(
    #         models=models,
    #         model_names=model_names,
    #         model_callbacks=model_callbacks,
    #         fit_callbacks=fit_callbacks,
    #         predict_callbacks=predict_callbacks,
    #         fit_args=fit_args
    #     )
    #
    #     self._set_score_batch(response_col=response_col,
    #                           predicted_col=predicted_col,
    #                           metrics=metrics,
    #                           include_steps=include_steps)