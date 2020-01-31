import pandas as pd
from orbit.utils.metrics import (
    smape,
    wmape
)
from orbit.exceptions import BacktestException
from orbit.utils.constants import (
    BacktestMetaKeys,
    BacktestFitColumnNames,
    BacktestAnalyzeKeys
)


class Backtest(object):
    """Object used to backtest..."""

    def __init__(self, df, min_train_len, incremental_len, forecast_len,
                 n_splits=None, scheme='expanding'):
        """Initializes Backtest object with DataFrame and splits data

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
        n_splits : int
            number of splits
        scheme : { 'expanding', 'rolling }
            split scheme

        """

        self.df = df.copy()

        self.scheme = scheme
        self.incremental_len = incremental_len
        self.forecast_len = forecast_len
        self.min_train_len = min_train_len
        self.n_splits = n_splits

        # defaults
        self._set_defaults()

        # validate
        self._validate_params()

        # init meta data
        self._bt_meta = {}

        # timeseries cross validation split
        self._set_split_idx()

        # init predictions df
        # grouped by model, split, time
        self._predicted_df = pd.DataFrame()

        # init results df
        # grouped by model, split
        self._score_df = pd.DataFrame()

        # batch model objects
        self._models = None
        self._model_names = None

    def _validate_params(self):
        if self.scheme not in ['expanding', 'rolling']:
            raise BacktestException('unknown scheme name...')

        # forecast length invalid
        if self.forecast_len <= 0:
            raise BacktestException('holdout period length must be positive...')

        # train + test length cannot be longer than df length
        if self.min_train_len + self.forecast_len > self._df_length:
            raise BacktestException('required time span is more than the full data frame...')

        if self.n_splits is not None and self.n_splits < 1:
            raise BacktestException('n_split must be a positive number')

    def _set_defaults(self):
        df = self.df

        self._df_length = df.shape[0]

        # if n_splits is specified, set min_train_len internally
        if self.n_splits:
            self.min_train_len = \
                self._df_length - self.forecast_len - (self.n_splits - 1) * self.incremental_len

    def _set_split_idx(self):
        scheme = self.scheme
        bt_end_min = self.min_train_len - 1
        bt_end_max = self._df_length - self.forecast_len
        bt_seq = range(bt_end_min, bt_end_max, self.incremental_len)

        bt_meta = {}
        for i, train_end_idx in enumerate(bt_seq):
            bt_meta[i] = {}
            train_start_idx = train_end_idx - self.min_train_len + 1 \
                if scheme == 'rolling' else 0
            bt_meta[i][BacktestMetaKeys.TRAIN_IDX.value] = range(train_start_idx, train_end_idx + 1)
            bt_meta[i][BacktestMetaKeys.TEST_IDX.value] = range(
                train_end_idx + 1, train_end_idx + self.forecast_len + 1)

        self._bt_meta = bt_meta

    def _fit(self, model, model_callback=None,
             fit_callback=None,predict_callback=None, fit_args=None):
        """Fits the splitted data to the model and predicts

        Parameters
        ----------
        model : object
            arbitrary instantiated model object
        model_callback : callable
            optional callback to adapt model object to work with `orbit.backtest`. Particularly,
            the object needs a `fit()` and `predict()` methods if not exists.
        fit_callback : callable
            optional callback to adapt data to work with `model`'s `fit()` method
        predict_callback : callable
            optional callback to adapt prediction results to work with `orbit.backtest`
        kwargs : args
            additional kwargs to be passed to the `fit_callback` function

        Returns
        -------
        pd.DataFrame
            DataFrame with Predictions

        """
        df = self.df
        bt_meta = self._bt_meta

        # todo: kwargs need to be parsed to know
        #   which callbacks they belong to
        #   alternatively, dict for each callback so we know clearly

        predicted_df = pd.DataFrame({})

        for meta_key, meta_value in bt_meta.items():
            # if we need to extend model to work with backtest
            if model_callback is not None:
                model = model_callback(model)

            train_df = df.iloc[meta_value[BacktestMetaKeys.TRAIN_IDX.value], :]\
                .reset_index(drop=True)
            test_df = df.iloc[meta_value[BacktestMetaKeys.TEST_IDX.value], :]\
                .reset_index(drop=True)

            # fit model
            if fit_callback is None:
                model.fit(train_df)
            else:
                fit_callback(model, train_df, **fit_args)

            # predict with model
            if predict_callback is None:
                predicted_out = model.predict(test_df)
            else:
                predicted_out = predict_callback(model, test_df, **fit_args)

            # predicted results should have same length as test_df
            if test_df.shape[0] != predicted_out.shape[0]:
                raise BacktestException('Prediction length should be the same as test df')

            results = pd.concat(
                (test_df, predicted_out), axis=1)

            # drop duplicate columns
            results = results.loc[:, ~results.columns.duplicated()]

            results['split_key'] = meta_key

            predicted_df = pd.concat(
                (predicted_df, results), axis=0)

        predicted_df = predicted_df.reset_index()
        predicted_df = predicted_df.rename(columns={'index': 'steps'})
        predicted_df['steps'] += 1
        self._predicted_df = predicted_df

        return self._predicted_df

    def _score_by(self, response_col, groupby, predicted_col='prediction', metrics=None):
        if metrics is None:
            metrics = {'wmape': wmape, 'smape': smape}

        predicted_df = self.get_predictions()
        score_df = pd.DataFrame({})

        # multiple models or step segmentation
        if groupby is not None:
            for metric_name, metric_fun in metrics.items():
                score_df[metric_name] = predicted_df \
                    .groupby(by=groupby) \
                    .apply(lambda x: metric_fun(x[response_col], x[predicted_col]))
            score_df = score_df.reset_index()

        # aggregate without groups
        else:
            for metric_name, metric_fun in metrics.items():
                score_df[metric_name] = pd.Series(
                    metric_fun(predicted_df[response_col], predicted_df[predicted_col])
                )
            score_df = score_df.reset_index(drop=True)

        return score_df

    def _set_score(self, response_col, predicted_col='prediction',
                   metrics=None, include_steps=False):

        groupby = ['steps'] if include_steps else None

        score_df = self._score_by(
            response_col=response_col,
            groupby=groupby,
            predicted_col=predicted_col,
            metrics=metrics
        )

        self._score_df = score_df

    def _fit_batch(self, models, model_names=None, model_callbacks=None, fit_callbacks=None,
                   predict_callbacks=None, fit_args=None):
        """Runs `_fit()` on a batch of models

        Parameters
        ----------
        models
        model_callbacks
        fit_callbacks
        predict_callbacks
        kwargs

        Returns
        -------

        """
        # store model object if we need to retrieve later
        self._models = models
        self._model_names = model_names

        batch_size = len(models)

        if model_callbacks is None:
            model_callbacks = [None] * batch_size
        if fit_callbacks is None:
            fit_callbacks = [None] * batch_size
        if predict_callbacks is None:
            predict_callbacks = [None] * batch_size
        if fit_args is None:
            fit_args = [None] * batch_size

        # todo: validate that all the lengths of args align

        predicted_df = self._predicted_df

        for idx, model in enumerate(models):
            each_predicted_df = self._fit(
                model,
                model_callback=model_callbacks[idx],
                fit_callback=fit_callbacks[idx],
                predict_callback=predict_callbacks[idx],
                fit_args=fit_args[idx]
            )  # note this also sets `self._predicted_df` inside `_fit()`

            # to retrieve from `self._models` if needed
            each_predicted_df['model_idx'] = idx

            # some model types raise errors if not converted to string here
            # e.g. models with `len()` over-ridden
            if model_names:
                each_predicted_df['model'] = model_names[idx]
            else:
                each_predicted_df['model'] = model.__class__.__name__

            predicted_df = pd.concat((predicted_df, each_predicted_df), axis=0, ignore_index=True)

        self._predicted_df = predicted_df.reset_index(drop=True)

    def _set_score_batch(self, response_col, predicted_col='prediction',
                         metrics=None, include_steps=True):

        # groupby determined by `include_steps` bool
        groupby = ['model_idx', 'model', 'steps'] \
            if include_steps \
            else ['model_idx', 'model']

        score_df = self._score_by(
            response_col=response_col,
            groupby=groupby,
            predicted_col=predicted_col,
            metrics=metrics
        )

        self._score_df = score_df

    def fit_score(self, model, response_col, predicted_col='prediction', metrics=None,
                  include_steps=False, model_callback=None, fit_callback=None,
                  predict_callback=None, fit_args=None):
        if fit_args is None:
            fit_args = {}

        self._fit(
            model=model,
            model_callback=model_callback,
            fit_callback=fit_callback,
            predict_callback=predict_callback,
            fit_args=fit_args
        )

        self._set_score(response_col=response_col, predicted_col=predicted_col,
                        metrics=metrics, include_steps=include_steps)

    def fit_score_batch(self, models, response_col, predicted_col='prediction', metrics=None,
                        model_names=None, include_steps=False, model_callbacks=None,
                        fit_callbacks=None, predict_callbacks=None, fit_args=None):

        self._fit_batch(
            models=models,
            model_names=model_names,
            model_callbacks=model_callbacks,
            fit_callbacks=fit_callbacks,
            predict_callbacks=predict_callbacks,
            fit_args=fit_args
        )

        self._set_score_batch(response_col=response_col,
                              predicted_col=predicted_col,
                              metrics=metrics,
                              include_steps=include_steps)

    def _append_split_meta(self):
        pass

    def _append_model_meta(self):
        pass

    def get_scores(self, include_model_meta=False):
        return self._score_df

    def get_predictions(self, include_split_meta=False):
        return self._predicted_df

    def get_meta(self):
        return self._bt_meta
