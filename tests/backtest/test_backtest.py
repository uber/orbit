import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from orbit.backtest.backtest import Backtest
from orbit.lgt import LGT
from orbit.dlt import DLT


def test_single_model_score(iclaims_training_data):
    bt = Backtest(
        iclaims_training_data,
        min_train_len=150,
        incremental_len=13,
        forecast_len=13,
        n_splits=4
    )

    lgt = LGT(
        response_col='claims',
        date_col='week',
        predict_method='mean',
        sample_method='vi',
        seasonality=52,
        chains=4,
    )

    # fit and score the model
    bt.fit_score(lgt, response_col='claims', predicted_col='prediction')

    # retrieve predictions
    predictions_df = bt.get_predictions()

    # retrieve scores
    scores_df = bt.get_scores()

    # todo: unit test assertions
    assert True


def test_single_model_score_with_callback(iclaims_training_data):
    # backtest instantiated the same regardless of model
    bt = Backtest(
        iclaims_training_data,
        min_train_len=150,
        incremental_len=13,
        forecast_len=13,
        n_splits=4
    )

    # sklearn model
    rf = RandomForestRegressor(n_estimators=50)

    # custom callback
    def fit_callback_sklearn(model, train_df, response_col, regressor_col):
        y = train_df[response_col]
        X = train_df[regressor_col]
        model.fit(X, y)
        return

    def predict_callback_sklearn(model, test_df, response_col, regressor_col):
        X = test_df[regressor_col]
        pred = model.predict(X)

        return pd.DataFrame(pred, columns=['prediction'])

    fit_args = {
        'response_col': 'claims',
        'regressor_col': ['trend.unemploy', 'trend.filling', 'trend.job']
    }

    bt.fit_score(
        rf,
        response_col='claims',
        predicted_col='prediction',
        fit_callback=fit_callback_sklearn,
        predict_callback=predict_callback_sklearn,
        fit_args=fit_args
    )

    predictions_df = bt.get_predictions()
    scores_df = bt.get_scores()

    # todo: write assertions
    assert True


def test_batch_model_score(iclaims_training_data):
    bt = Backtest(
        iclaims_training_data,
        min_train_len=150,
        incremental_len=13,
        forecast_len=13,
        n_splits=4
    )

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        sample_method='vi',
    )

    dlt = DLT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        sample_method='vi',
    )

    # batch fit and score models
    bt.fit_score_batch(models=[lgt, dlt], response_col='claims', predicted_col='prediction')

    # same way to retrieve info
    # results are still stored in a single object / dataframe
    predictions_df = bt.get_predictions()
    scores_df = bt.get_scores()

    # todo: write assertions
    assert True


def test_batch_model_score_with_callback(iclaims_training_data):
    bt = Backtest(
        iclaims_training_data,
        min_train_len=150,
        incremental_len=13,
        forecast_len=13,
        n_splits=4
    )

    lgt = LGT(
        response_col='claims',
        date_col='week',
        seasonality=52,
        chains=4,
        predict_method='mean',
        sample_method='vi',
    )

    # sklearn model
    rf = RandomForestRegressor(n_estimators=50)

    # custom callback
    def fit_callback_sklearn(model, train_df, response_col, regressor_col):
        y = train_df[response_col]
        X = train_df[regressor_col]
        model.fit(X, y)
        return

    def predict_callback_sklearn(model, test_df, response_col, regressor_col):
        X = test_df[regressor_col]
        pred = model.predict(X)

        return pd.DataFrame(pred, columns=['prediction'])

    def predict_callback_lgt(model, test_df):
        pred = model.predict(test_df)
        pred.drop(columns=['week'], inplace=True)

        return pred

    fit_args = {
        'response_col': 'claims',
        'regressor_col': ['trend.unemploy', 'trend.filling', 'trend.job']
    }

    # batch fit and score models
    bt.fit_score_batch(
        models=[lgt, rf],
        response_col='claims',
        predicted_col='prediction',
        fit_callbacks=[None, fit_callback_sklearn],
        predict_callbacks=[predict_callback_lgt, predict_callback_sklearn],
        fit_args=[{}, fit_args]
    )

    # same way to retrieve info
    # results are still stored in a single object / dataframe
    predictions_df = bt.get_predictions()
    scores_df = bt.get_scores()

    # todo: write assertions
    assert True
