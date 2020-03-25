import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from orbit.backtest.backtest import TimeSeriesSplitter, Backtest
from orbit.lgt import LGT


def test_single_model_score(iclaims_training_data):
    splitter = TimeSeriesSplitter(
        iclaims_training_data,
        min_train_len=150,
        incremental_len=12,
        forecast_len=8,
        n_splits=4
    )

    bt = Backtest(
        splitter=splitter
    )

    lgt = LGT(
        response_col='claims',
        date_col='week',
        predict_method='mean',
        sample_method='vi',
        seasonality=52,
        chains=4,
    )

    # check meta data
    split_scheme = splitter.get_scheme()

    first_run = split_scheme.get(0)
    last_run = split_scheme.get(3)

    # fit and score the model
    bt.fit_score(lgt, response_col='claims', predicted_col='prediction')

    # retrieve predictions
    predictions_df = bt.get_predictions()

    # retrieve scores
    scores_df = bt.get_scores()

    expected_predict_df_columns = ['steps', 'week', 'claims','trend.unemploy',
                                   'trend.filling', 'trend.job', 'prediction', 'split_key']
    expected_predict_df_shapes = (32, 8)
    expected_scores_df_shape = (1, 2)
    expected_splits = 4
    expected_first_train_idx = range(0, 399)
    expected_first_test_idx = range(399, 407)
    expected_last_train_idx = range(0, 435)
    expected_last_test_idx = range(435, 443)

    assert list(predictions_df.columns) == expected_predict_df_columns
    assert len(split_scheme) == expected_splits
    assert first_run.get('train_idx') == expected_first_train_idx
    assert first_run.get('test_idx') == expected_first_test_idx
    assert last_run.get('train_idx') == expected_last_train_idx
    assert last_run.get('test_idx') == expected_last_test_idx
    assert predictions_df.shape == expected_predict_df_shapes
    assert scores_df.shape == expected_scores_df_shape

    # todo: unit tests with real values


def test_single_model_score_with_callback(iclaims_training_data):
    # backtest instantiated the same regardless of model
    splitter = TimeSeriesSplitter(
        iclaims_training_data,
        min_train_len=150,
        incremental_len=13,
        forecast_len=13,
        n_splits=4
    )

    bt = Backtest(
        splitter=splitter
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

    fit_predict_args = {
        'response_col': 'claims',
        'regressor_col': ['trend.unemploy', 'trend.filling', 'trend.job']
    }

    bt.fit_score(
        rf,
        response_col='claims',
        predicted_col='prediction',
        fit_callback=fit_callback_sklearn,
        predict_callback=predict_callback_sklearn,
        fit_args=fit_predict_args,
        predict_args=fit_predict_args
    )

    predictions_df = bt.get_predictions()
    scores_df = bt.get_scores()

    expected_predict_df_columns = ['steps', 'week', 'claims', 'trend.unemploy',
                                   'trend.filling', 'trend.job', 'prediction', 'split_key']
    expected_predict_df_shapes = (52, 8)
    expected_scores_df_shape = (1, 2)

    assert list(predictions_df.columns) == expected_predict_df_columns
    assert predictions_df.shape == expected_predict_df_shapes
    assert scores_df.shape == expected_scores_df_shape

    # todo: unit tests with real values


# def test_batch_model_score(iclaims_training_data):
#     bt = Backtest(
#         iclaims_training_data,
#         min_train_len=150,
#         incremental_len=13,
#         forecast_len=13,
#         n_splits=4
#     )
#
#     lgt = LGT(
#         response_col='claims',
#         date_col='week',
#         seasonality=52,
#         chains=4,
#         predict_method='mean',
#         sample_method='vi',
#     )
#
#     dlt = DLT(
#         response_col='claims',
#         date_col='week',
#         seasonality=52,
#         chains=4,
#         predict_method='mean',
#         sample_method='vi',
#     )
#
#     # batch fit and score models
#     bt.fit_score_batch(models=[lgt, dlt], response_col='claims', predicted_col='prediction')
#
#     # same way to retrieve info
#     # results are still stored in a single object / dataframe
#     predictions_df = bt.get_predictions()
#     scores_df = bt.get_scores()
#
#     expected_predict_df_columns = ['steps', 'week', 'claims', 'trend.unemploy',
#                                    'trend.filling', 'trend.job', 'prediction', 'split_key',
#                                    'model_idx', 'model']
#     expected_predict_df_shapes = (104, 10)
#     expected_scores_df_shape = (2, 4)
#     expected_number_of_models = 2
#
#     assert list(predictions_df.columns) == expected_predict_df_columns
#     assert predictions_df.shape == expected_predict_df_shapes
#     assert scores_df.shape == expected_scores_df_shape
#     assert len(predictions_df['model_idx'].unique()) == expected_number_of_models
#
#     # todo: write with real values
#
#
# def test_batch_model_score_with_callback(iclaims_training_data):
#     bt = Backtest(
#         iclaims_training_data,
#         min_train_len=150,
#         incremental_len=13,
#         forecast_len=13,
#         n_splits=4
#     )
#
#     lgt = LGT(
#         response_col='claims',
#         date_col='week',
#         seasonality=52,
#         chains=4,
#         predict_method='mean',
#         sample_method='vi',
#     )
#
#     # sklearn model
#     rf = RandomForestRegressor(n_estimators=50)
#
#     # custom callback
#     def fit_callback_sklearn(model, train_df, response_col, regressor_col):
#         y = train_df[response_col]
#         X = train_df[regressor_col]
#         model.fit(X, y)
#         return
#
#     def predict_callback_sklearn(model, test_df, response_col, regressor_col):
#         X = test_df[regressor_col]
#         pred = model.predict(X)
#
#         return pd.DataFrame(pred, columns=['prediction'])
#
#     fit_args = {
#         'response_col': 'claims',
#         'regressor_col': ['trend.unemploy', 'trend.filling', 'trend.job']
#     }
#
#     # batch fit and score models
#     bt.fit_score_batch(
#         models=[lgt, rf],
#         response_col='claims',
#         predicted_col='prediction',
#         fit_callbacks=[None, fit_callback_sklearn],
#         predict_callbacks=[None, predict_callback_sklearn],
#         fit_args=[{}, fit_args]
#     )
#
#     # same way to retrieve info
#     # results are still stored in a single object / dataframe
#     predictions_df = bt.get_predictions()
#     scores_df = bt.get_scores()
#
#     # todo: write assertions
#     assert True


# def test_single_model_score_with_steps(iclaims_training_data):
#     bt = Backtest(
#         iclaims_training_data,
#         min_train_len=150,
#         incremental_len=13,
#         forecast_len=13,
#         n_splits=4
#     )
#
#     lgt = LGT(
#         response_col='claims',
#         date_col='week',
#         predict_method='mean',
#         sample_method='vi',
#         seasonality=52,
#         chains=4,
#     )
#
#     # fit and score the model
#     bt.fit_score(lgt, response_col='claims', predicted_col='prediction', include_steps=True)
#
#     # retrieve predictions
#     predictions_df = bt.get_predictions()
#
#     # retrieve scores
#     scores_df = bt.get_scores()
#
#     expected_predict_df_columns = ['steps', 'week', 'claims','trend.unemploy',
#                                    'trend.filling', 'trend.job', 'prediction', 'split_key']
#     expected_predict_df_shapes = (52, 8)
#     expected_scores_df_shape = (13, 3)
#
#     assert list(predictions_df.columns) == expected_predict_df_columns
#     assert predictions_df.shape == expected_predict_df_shapes
#     assert scores_df.shape == expected_scores_df_shape
#
#     # todo: write with more real prediction values
