import pandas as pd
import numpy as np
import tqdm
import pickle
import os
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from orbit.exceptions import BacktestException
from orbit.utils.constants import (
    BacktestMetaKeys,
    BacktestFitColumnNames,
    BacktestAnalyzeKeys
)
from orbit.utils.utils import is_ordered_datetime


# a few utility functions to calculate mape/wmape/smape metrics
def mape(actual, pred, transform = None):
    ''' calculate mape metric

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        mape value
    '''
    actual = np.array(actual)
    pred = np.array(pred)
    if transform is not None:
        actual = transform(actual)
        pred = transform(pred)
    return np.mean(np.abs( (actual - pred) / actual ))


def wmape(actual, pred, transform = None):
    ''' calculate weighted mape metric

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        wmape value
    '''
    actual = np.array(actual)
    pred = np.array(pred)
    if transform is not None:
        actual = transform(actual)
        pred = transform(pred)
    weights = np.abs(actual) / np.sum(np.abs(actual))
    return np.sum( weights * np.abs( (actual - pred) / actual ) )


def smape(actual, pred, transform = None):
    ''' calculate symmetric mape

    Parameters
    ----------
    actual: array-like
        true values
    pred: array-like
        prediction values

    Returns
    -------
    float
        symmetric mape value
    '''
    actual = np.array(actual)
    pred = np.array(pred)
    if transform is not None:
        actual = transform(actual)
        pred = transform(pred)
    return np.mean(2*np.abs(pred - actual)/(np.abs(actual) + np.abs(pred)))


class BacktestEngine:
    ''' Back-testing engine for an existing model object; currently, it only supports
    univariate time series.

    Parameters
    ----------
    model: initialized model object
        Mobel object to be back-tested; default setting works for Orbit model object.
        One can write customized callbacks to accommodate any model object which has
        .fit and .predict methods.
    df: pandas data frame
        input data set
    model_callbacks: optional function; default to be None
        add .get_params() method to model object instance such that new model object can be created with
        the same initializations
    kwargs:
        optional parameters to specify `date_col` and `response_col` for model objects which
        don't have such attributes


    Attributes
    ----------
    bt_meta: dictionary
        meta data storing the info for each back-testing period/model
    bt_res: pandas data frame
        actuals/predictions/horizons of each back-testing model
    analyze_res: dictionary
        aggregated metric results

    '''
    def __init__(self, model, df, model_callbacks=None, **kwargs):

        self.model = model_callbacks(model) if model_callbacks is not None else model
        self.df = df.copy()
        for key in ['date_col', 'response_col']:
            setattr(self, key, kwargs[key]) if key in kwargs.keys() else \
                setattr(self, key, getattr(model, key))

        if not is_ordered_datetime(df[self.date_col]):
            raise BacktestException('Datetime index must be ordered and not repeat...')

    def create_meta(self, min_train_len, incremental_len, forecast_len, start_date=None, end_date=None,
                    keep_cols=None,  scheme='expanding'):
        """create meta data for back-testing based on the scheduling related parameters.

        Parameters
        ----------
        min_train_len: integer
            the minimum number of observations required for the training period
        incremental_len: integer
            the number of observations between each successive backtest period
        forecast_len: integer
            forecast length
        start_date: string
            optional parameter used to filter input data set; default to be the min date in data
        end_date: string
            optional parameter used to filter input data set; default to be the max date in data
        keep_cols: list of strings
            columns to be kept in the back-testing results
        scheme: string
            'expanding' or 'rolling'


        """
        self.scheme = scheme
        self.min_train_len = min_train_len
        self.incremental_len = incremental_len
        self.forecast_len = forecast_len
        self.keep_cols = keep_cols

        df = self.df
        date_col = self.date_col
        start_date = start_date if start_date is not None else min(df[date_col])
        end_date = end_date if end_date is not None else max(df[date_col])
        self.start_date = start_date
        self.end_date = end_date

        start_date_idx = np.where(df[date_col] == start_date)[0][0]
        end_date_idx = np.where(df[date_col] == end_date)[0][0]

        if min_train_len + forecast_len > end_date_idx - start_date_idx + 1:
            raise BacktestException('required time span is more than the full data frame...')
        if forecast_len <= 0:
            raise BacktestException('holdout period length must be positive...')
        if scheme not in ['expanding', 'rolling']:
            raise BacktestException('unknown scheme name...')

        # it's more robust to deal with idx instead of dates?
        bt_end_min = start_date_idx + min_train_len - 1
        bt_end_max = end_date_idx - forecast_len
        bt_seq = range(bt_end_min, bt_end_max + 1, incremental_len)

        bt_meta = {}
        for i, train_end_idx in enumerate(bt_seq):
            bt_meta[i] = {}
            bt_meta[i][BacktestMetaKeys.MODEL.value] = self.model.__class__(**self.model.get_params())
            train_start_idx = train_end_idx - min_train_len + 1 if scheme == 'rolling' else start_date_idx
            bt_meta[i][BacktestMetaKeys.TRAIN_START_DATE.value] = df[date_col].iloc[train_start_idx]
            bt_meta[i][BacktestMetaKeys.TRAIN_END_DATE.value] = df[date_col].iloc[train_end_idx]
            bt_meta[i][BacktestMetaKeys.TRAIN_IDX.value] = range(train_start_idx, train_end_idx + 1)
            bt_meta[i][BacktestMetaKeys.TEST_IDX.value] = range(
                train_end_idx+1, train_end_idx + forecast_len + 1)
            bt_meta[i][BacktestMetaKeys.FORECAST_DATES.value] = df[date_col].iloc[range(
                train_end_idx+1, train_end_idx + forecast_len + 1)] # one row less

        self.bt_meta = bt_meta

    def run(self, verbose=True, save_results=False, pred_col='prediction',
            fit_callbacks=None, pred_callbacks=None, **kwargs):
        ''' Run the back-testing procedures based on the meta data

        Parameters
        -----------
        pred_col: string
            the column name to extract the predictions; for example, 'prediction' for Orbit model
            object with MAP as the predict_method
        verbose: logical
            if printing out the detailed training/forecasting dates info
        save_results: logical
            if save back-testing results into disk
        fit_callbacks: optional function
            callback function for the fit mode
        pred_callbacks: optional function
            callback function for the predict mode
        kwargs: additional parameters for callback functions

        '''

        col_names = [col.value for col in BacktestFitColumnNames]
        col_names = col_names + self.keep_cols if self.keep_cols is not None else col_names
        bt_res = pd.DataFrame({}, columns=col_names)
        df = self.df
        if verbose:
            print('run {} window back-testing:'.format(self.scheme))
        for i in tqdm.tqdm(self.bt_meta.keys()):
            forecast_dates = self.bt_meta[i][BacktestMetaKeys.FORECAST_DATES.value]
            if verbose:
                print(
                    'training and forcasting for horizon {} -- {}'.format(
                        forecast_dates.iloc[0].strftime('%m/%d/%Y'),
                        forecast_dates.iloc[-1].strftime('%m/%d/%Y')))
            model = self.bt_meta[i][BacktestMetaKeys.MODEL.value]
            train_df = df.iloc[self.bt_meta[i][BacktestMetaKeys.TRAIN_IDX.value], :]
            test_df = df.iloc[self.bt_meta[i][BacktestMetaKeys.TEST_IDX.value], :]
            if fit_callbacks is None:
                model.fit(train_df)
            else:
                fit_callbacks(model, train_df, **kwargs)

            if pred_callbacks is None:
                pred_res = model.predict(test_df)
            else:
                pred_res = pred_callbacks(model, test_df, **kwargs)

            pred = pred_res[pred_col].values if pred_col is not None else pred_res
            actual = test_df[self.response_col].values

            res = pd.DataFrame({
                BacktestFitColumnNames.TRAIN_START_DATE.value: self.bt_meta[i][
                    BacktestMetaKeys.TRAIN_START_DATE.value],
                BacktestFitColumnNames.TRAIN_END_DATE.value: self.bt_meta[i][
                    BacktestMetaKeys.TRAIN_END_DATE.value],
                BacktestFitColumnNames.FORECAST_DATES.value: forecast_dates,
                BacktestFitColumnNames.ACTUAL.value: actual,
                BacktestFitColumnNames.PRED.value: pred,
                BacktestFitColumnNames.PRED_HORIZON.value: np.arange(1, len(forecast_dates) + 1)
            })
            if self.keep_cols is not None:
                res = pd.concat([
                    res, df.iloc[
                         self.bt_meta[i][BacktestMetaKeys.TEST_IDX.value], :][self.keep_cols]
                ], axis=1)

            bt_res = pd.concat([bt_res, res], axis=0, ignore_index=True)

        if save_results:
            now = dt.datetime.now()
            outdir = os.getcwd() + '/backtest_results'
            os.makedirs(outdir, exist_ok=True)
            outfile = outdir + '/backtest_output_' + now.strftime("%Y%m%d%H%M") + '.pickle'
            bt_out = {
                'bt_meta': self.bt_meta,
                'bt_res': bt_res
            }
            print('saving results to', outfile)
            with open(outfile, 'wb') as handle:
                pickle.dump(bt_out, handle)
        self.bt_res = bt_res

        return

    def analyze(self, metrics='mape', transform=None):
        ''' utility function to calculate the aggregated metrics based on the back-testing runs

        Parameters
        ----------
        metrics: string
            metric name, 'mape', 'wmape', or 'smape';
        transform:
            tranform function applied to actuals/predictions; default to be None.

        '''
        if self.bt_res is None:
            raise BacktestException("run .run() before analyzing...")
        metric_per_btmod = self.bt_res.groupby(BacktestFitColumnNames.TRAIN_END_DATE.value).apply(
            lambda x: eval(metrics)(x[BacktestFitColumnNames.ACTUAL.value],
                                    x[BacktestFitColumnNames.PRED.value],
                                    transform=transform))
        metric_geo = eval(metrics)(self.bt_res[BacktestFitColumnNames.ACTUAL.value],
                                   self.bt_res[BacktestFitColumnNames.PRED.value],
                                   transform=transform)
        metric_per_horizon = self.bt_res.groupby(BacktestFitColumnNames.PRED_HORIZON.value).apply(
            lambda x: eval(metrics)(x[BacktestFitColumnNames.ACTUAL.value],
                                    x[BacktestFitColumnNames.PRED.value],
                                    transform=transform))

        analyze_res = {
            BacktestAnalyzeKeys.METRIC_NAME.value: metrics,
            BacktestAnalyzeKeys.METRIC_PER_BTMOD.value: metric_per_btmod,
            BacktestAnalyzeKeys.METRIC_GEO.value: metric_geo,
            BacktestAnalyzeKeys.METRIC_PER_HORIZON.value: metric_per_horizon}
        self.analyze_res = analyze_res

        return

    def plot_horizon(self, **kwargs):
        ''' plotting function: metrics vs horizon
        '''
        if self.analyze_res is None:
            raise BacktestException("run .analyze() before plotting...")
        fig, ax = plt.subplots(**kwargs)
        ax.plot(self.analyze_res[BacktestAnalyzeKeys.METRIC_PER_HORIZON.value])

        ax.set_title('{} window back-testing over horizon'.format(self.scheme))
        ax.set_xlabel('horizon')
        ax.set_ylabel(self.analyze_res[BacktestAnalyzeKeys.METRIC_NAME.value])

        return fig


def run_group_backtest(data, date_col, response_col, key_col, pred_cols,
                       mod_list, model_callbacks, fit_callbacks, pred_callbacks,
                       min_train_len, incremental_len, forecast_len,
                       transform_fun=None, start_date=None, end_date=None,
                       keep_cols=None, regressor_col=None, mod_names=None,
                       scheme='expanding'):

    metric_dict = {}
    assert len(mod_list) == len(
        pred_cols), 'check if the model list is consistent with the prediction columns'
    if mod_names is not None:
        assert len(mod_list) == len(mod_names)
    unique_keys = data[key_col].unique()
    all_res = []
    for i, mod in enumerate(mod_list):
        idxer = i if mod_names is None else mod_names[i]
        metric_dict[idxer] = {}
        res = []
        tic = time.time()
        for key in unique_keys:
            df = data[data[key_col] == key]
            bt_expand = BacktestEngine(mod, df, date_col=date_col, response_col=response_col,
                                       model_callbacks=model_callbacks[i])

            bt_expand.create_meta(min_train_len, incremental_len, forecast_len,
                                  start_date=start_date, end_date=end_date, keep_cols=keep_cols,
                                  scheme=scheme)

            bt_expand.run(verbose=False, save_results=False,
                          fit_callbacks=fit_callbacks[i], pred_callbacks=pred_callbacks[i],
                          pred_col=pred_cols[i],
                          date_col=date_col, response_col=response_col, regressor_col=regressor_col)
            tmp = bt_expand.bt_res.copy()
            res.append(tmp)

        res = pd.concat(res, axis=0, ignore_index=True)
        if mod_names is not None:
            res['model'] = mod_names[i]
        all_res.append(res)
        if transform_fun is not None:
            res[['actual', 'pred']] = res[['actual', 'pred']].apply(transform_fun)
        toc = time.time()
        print('time elapsed {}'.format(time.strftime("%H:%M:%S", time.gmtime(toc - tic))))
    all_res = pd.concat(all_res, axis=0, ignore_index=True)
    return all_res


