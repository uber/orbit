import custom_inherit as ci
from copy import copy, deepcopy
import numpy as np
import pandas as pd

from ..constants.constants import PredictMethod, PredictionKeys
from ..estimators.stan_estimator import StanEstimatorMCMC
from ..utils.docstring_style import merge_numpy_docs_dedup
from ..utils.predictions import prepend_date_column, compute_percentiles
from ..exceptions import IllegalArgument, ModelException, PredictionException, AbstractMethodException
from ..utils.general import is_ordered_datetime


ci.store["numpy_with_merge_dedup"] = merge_numpy_docs_dedup
ci.add_style("numpy_with_merge_dedup", merge_numpy_docs_dedup)


class BaseTemplate(object, metaclass=ci.DocInheritMeta(style="numpy_with_merge_dedup")):
    """ Base abstract class for univariate time-series model creation
    `BaseTemplate` will instantiate an estimator class of `estimator_type`.
    Each model defines its own `_supported_estimator_types` to determine if
    the provided `estimator_type` is supported for that particular model.
    Parameters
    ----------
    response_col : str
        Name of response variable column, default 'y'
    date_col : str
        Name of date variable column, default 'ds'
    estimator_type : orbit.BaseEstimator
        Any subclass of `orbit.BaseEstimator`
    Notes
    -----
    For attributes which are input by users and needed to mutate further downstream, we will introduce a
    new internal attribute with identical name except a prefix "_".
    e.g. If x appear in the arg default as `None` and we need to impute by 0. we will have self._x = 0 downstream.
    """
    # TODO: for now we assume has to be ENUM, maybe we can allow list as well? such that left and right are always
    # using same name
    # data labels for sampler
    _data_input_mapper = None
    # used to match name of `*.stan` or `*.pyro` file to look for the model
    _model_name = None
    # TODO: right now we assume _fitter is only for one specific estimator type
    # TODO: in the future, we should make it for example like a dict {PyroEstimator: pyro_model} etc. in case we want
    # TODO: to support multiple estimators and use for validation
    # EXPERIMENTAL: _fitter is used for quick supply of pyro / stan object instead of supplying a file
    _fitter = None
    # supported estimators in ..estimators
    # concrete classes should overwrite this
    _supported_estimator_types = None  # set for each model

    def __init__(self, response_col='y', date_col='ds', estimator_type=StanEstimatorMCMC, **kwargs):
        # general fields passed into Base Template
        self.response_col = response_col
        self.date_col = date_col

        # basic response fields
        # mainly set by ._set_training_df_meta() and ._set_dynamic_attributes()
        self.response = None
        self.date_array = None
        self.num_of_observations = None
        self.training_start = None
        self.training_end = None
        self._model_data_input = None

        # basic estimator fields
        self.estimator_type = estimator_type
        self.estimator = self.estimator_type(**kwargs)
        self.with_mcmc = None
        # set by ._set_init_values
        # this is ONLY used by stan which by default used 'random'
        self._init_values = None

        self._validate_supported_estimator_type()
        self._set_with_mcmc()

        # set by _set_model_param_names()
        self._model_param_names = list()

        # set by `fit()`
        self._posterior_samples = dict()
        # init aggregate posteriors
        self._aggregate_posteriors = dict()

        # for full Bayesian, user can store full prediction array if requested
        self.prediction_array = None
        self.prediction_input_meta = dict()

        # storing metrics in training result meta
        self._training_metrics = dict()

    # initialization related modules
    def _validate_supported_estimator_type(self):
        if self.estimator_type not in self._supported_estimator_types:
            msg_template = "Model class: {} is incompatible with Estimator: {}.  Estimator Support: {}"
            model_class = type(self)
            estimator_type = self.estimator_type
            raise IllegalArgument(
                msg_template.format(model_class, estimator_type, str(self._supported_estimator_types))
            )

    def _set_with_mcmc(self):
        """Include extra indicator to indicate whether the object is using mcmc type of estimator
        """
        estimator_type = self.estimator_type
        # set `with_mcmc` attribute based on estimator type
        # if no attribute for _is_mcmc_estimator, default to False
        if getattr(estimator_type, '_is_mcmc_estimator', False):
            self.with_mcmc = 1
        else:
            self.with_mcmc = 0

    def _set_model_param_names(self, **kwargs):
        """Set label for model parameters.  This function can be dependent on
        static attributes.
        """
        raise AbstractMethodException(
            "Abstract method.  Model should implement concrete ._set_model_param_names().")

    def get_model_param_names(self):
        return self._model_param_names

    def _set_static_attributes(self, **kwargs):
        """Set static attributes which are independent from data matrix.
        These methods are supposed to be over-ride by child (model) template.
        For attributes dependent on data matrix, use _set_dynamic_attributes
        """
        pass

    # fit and predict related modules
    def _validate_training_df(self, df):
        df_columns = df.columns

        # validate date_col
        if self.date_col not in df_columns:
            raise ModelException("DataFrame does not contain `date_col`: {}".format(self.date_col))

        # validate ordering of time series
        date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        if not is_ordered_datetime(date_array):
            raise ModelException('Datetime index must be ordered and not repeat')

        # validate response variable is in df
        if self.response_col not in df_columns:
            raise ModelException("DataFrame does not contain `response_col`: {}".format(self.response_col))

    def _set_training_df_meta(self, df):
        self.response = df[self.response_col].values
        self.date_array = pd.to_datetime(df[self.date_col]).reset_index(drop=True)
        self.num_of_observations = len(self.response)
        self.response_sd = np.nanstd(self.response)
        self.training_start = df[self.date_col].iloc[0]
        self.training_end = df[self.date_col].iloc[-1]

    def _set_model_data_input(self):
        """Collects data attributes into a dict for sampling/optimization api"""
        # refresh a clean dict
        data_inputs = dict()

        if not self._data_input_mapper:
            raise ModelException('Empty or invalid data_input_mapper')

        for key in self._data_input_mapper:
            # mapper keys in upper case; inputs in lower case
            key_lower = key.name.lower()
            input_value = getattr(self, key_lower, None)
            if input_value is None:
                raise ModelException('{} is missing from data input'.format(key_lower))
            if isinstance(input_value, bool):
                # stan accepts bool as int only
                input_value = int(input_value)
            data_inputs[key.value] = input_value

        self._model_data_input = data_inputs

    def get_model_data_input(self):
        return self._model_data_input

    def _set_init_values(self):
        """Set init as a callable (for Stan ONLY)
        See: https://pystan.readthedocs.io/en/latest/api.htm
        """
        pass

    def get_init_values(self):
        return self._init_values

    def is_fitted(self):
        # if either aggregate posterior and posterior_samples are non-empty, claim it as fitted model (true),
        # else false.
        if bool(self._posterior_samples):
            return True
        for key in self._aggregate_posteriors.keys():
            if bool(self._aggregate_posteriors[key]):
                return True
        return False

    def _set_dynamic_attributes(self, df):
        """Set required input based on input DataFrame, rather than at object instantiation"""
        pass

    def fit(self, df):
        """Fit model to data and set extracted posterior samples"""
        estimator = self.estimator
        model_name = self._model_name
        df = df.copy()

        # default set and validation of input data frame
        self._validate_training_df(df)
        self._set_training_df_meta(df)

        # customize module
        self._set_dynamic_attributes(df)

        # default process post attributes setting
        # _set_model_data_input() behavior depends on _set_training_df_meta()
        self._set_model_data_input()
        # set initial values for randomization; right now only used by pystan; default as 'random'
        self._set_init_values()

        # estimator inputs
        data_input = self.get_model_data_input()
        init_values = self.get_init_values()
        model_param_names = self.get_model_param_names()

        # note that estimator will search for the .stan, .pyro model file based on the
        # estimator type and model_name provided
        model_extract, training_metrics = estimator.fit(
            model_name=model_name,
            model_param_names=model_param_names,
            data_input=data_input,
            fitter=self._fitter,
            init_values=init_values
        )

        self._posterior_samples = model_extract
        self._training_metrics = training_metrics

    def get_prediction_df_meta(self, df):
        # get prediction df meta
        prediction_df_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'prediction_start': df[self.date_col].iloc[0],
            'prediction_end': df[self.date_col].iloc[-1]
        }

        if not is_ordered_datetime(prediction_df_meta['date_array']):
            raise IllegalArgument('Datetime index must be ordered and not repeat')

        # TODO: validate that all regressor columns are present, if any

        if prediction_df_meta['prediction_start'] < self.training_start:
            raise PredictionException('Prediction start must be after training start.')

        return prediction_df_meta

    def get_posterior_samples(self):
        return self._posterior_samples.copy()

    def get_training_metrics(self):
        return self._training_metrics.copy()

    def get_prediction_input_meta(self, df):
        # remove reference from original input
        df = df.copy()

        # get prediction df meta
        prediction_input_meta = {
            'date_array': pd.to_datetime(df[self.date_col]).reset_index(drop=True),
            'df_length': len(df.index),
            'prediction_start': df[self.date_col].iloc[0],
            'prediction_end': df[self.date_col].iloc[-1],
        }

        if not is_ordered_datetime(prediction_input_meta['date_array']):
            raise IllegalArgument('Datetime index must be ordered and not repeat')

        # TODO: validate that all regressor columns are present, if any

        if prediction_input_meta['prediction_start'] < self.training_start:
            raise PredictionException('Prediction start must be after training start.')

        trained_len = self.num_of_observations

        # If we cannot find a match of prediction range, assume prediction starts right after train
        # end
        if prediction_input_meta['prediction_start'] > self.training_end:
            forecast_dates = set(prediction_input_meta['date_array'])
            n_forecast_steps = len(forecast_dates)
            # time index for prediction start
            start = trained_len
        else:
            # compute how many steps to forecast
            forecast_dates = \
                set(prediction_input_meta['date_array']) - set(self.date_array)
            # check if prediction df is a subset of training df
            # e.g. "negative" forecast steps
            n_forecast_steps = len(forecast_dates) or - (
                len(set(self.date_array) - set(prediction_input_meta['date_array']))
            )
            # time index for prediction start
            start = pd.Index(
                self.date_array).get_loc(prediction_input_meta['prediction_start'])

        prediction_input_meta.update({
            'start': start,
            'n_forecast_steps': n_forecast_steps,
        })

        self.prediction_input_meta = prediction_input_meta

    def predict(self, df, **kwargs):
        """Predict interface for users"""
        raise AbstractMethodException("Abstract method.  Model should implement concrete .predict().")

    def _predict(self, posterior_estimates, df, include_error=False, **kwargs):
        """Inner predict being called internal for different prediction purpose such as bootstrapping"""
        raise AbstractMethodException("Abstract method.  Model should implement concrete ._predict().")


class MAPTemplate(BaseTemplate):
    """ Abstract class for MAP (Maximum a Posteriori) prediction
    In this module, prediction is based on Maximum a Posteriori (aka Mode) of the posterior.
    This template only supports MAP inference.
    Parameters
    ----------
    n_bootstrap_draws : int
        Number of bootstrap samples to draw from the error part to generate the uncertainty.
        If set to be -1, will use the original posterior draw (no uncertainty).
    prediction_percentiles : list
        List of integers of prediction percentiles that should be returned on prediction. To avoid reporting any
        confident intervals, pass an empty list
    """

    def __init__(self, n_bootstrap_draws=1e4, prediction_percentiles=None, **kwargs):
        super().__init__(**kwargs)

        # n_bootstrap_draws here only to provide empirical prediction percentiles;
        # mid-point estimate is always replaced
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles
        self._prediction_percentiles = None

        # unlike full prediction, it does not take negative number of bootstrap draw
        # if self.n_bootstrap_draws < 2:
        #     raise IllegalArgument("Error: Number of bootstrap draws must be at least 2")
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles))  # unique set
        self._prediction_percentiles.sort()

        # override init aggregate posteriors
        self._aggregate_posteriors = {PredictMethod.MAP.value: dict()}

        self._set_static_attributes()
        self._set_model_param_names()

    # fit and predict related modules
    def _set_map_posterior(self):
        """ set MAP posteriors with right dimension"""
        posterior_samples = self._posterior_samples
        map_posterior = {}
        for param_name in self._model_param_names:
            param_array = posterior_samples[param_name]
            # add dimension so it works with vector math in `_predict`
            param_array = np.expand_dims(param_array, axis=0)
            map_posterior.update({param_name: param_array})

        self._aggregate_posteriors[PredictMethod.MAP.value] = map_posterior

    def fit(self, df):
        super().fit(df)
        self._set_map_posterior()

    def predict(self, df, decompose=False, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self.get_prediction_input_meta(df)

        # perform point prediction
        aggregate_posteriors = self._aggregate_posteriors.get(PredictMethod.MAP.value)
        point_predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors, df=df, include_error=False, **kwargs
        )
        for k, v in point_predicted_dict.items():
            point_predicted_dict[k] = np.squeeze(v, 0)

        # to derive confidence interval; the condition should be sufficient since we add [50] by default
        if self.n_bootstrap_draws > 0 and len(self._prediction_percentiles) > 1:
            # perform bootstrap; we don't have posterior samples. hence, we just repeat the draw here.
            posterior_samples = {}
            for k, v in aggregate_posteriors.items():
                posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)
            predicted_dict = self._predict(
                posterior_estimates=posterior_samples, df=df, include_error=True, **kwargs
            )
            percentiles_dict = compute_percentiles(predicted_dict, self._prediction_percentiles)
            # replace mid point prediction by point estimate
            percentiles_dict.update(point_predicted_dict)

            if PredictionKeys.PREDICTION.value not in percentiles_dict.keys():
                raise PredictionException("cannot find the key:'{}' from return of _predict()".format(
                    PredictionKeys.PREDICTION.value))

            # reduce to prediction only if decompose is not requested
            if not decompose:
                k = PredictionKeys.PREDICTION.value
                reduced_keys = [k + "_" + str(p) if p != 50 else k for p in self._prediction_percentiles]
                percentiles_dict = {k: v for k, v in percentiles_dict.items() if k in reduced_keys}
            predicted_df = pd.DataFrame(percentiles_dict)
        else:
            # reduce to prediction only if decompose is not requested
            if not decompose:
                point_predicted_dict = {
                    k: v for k, v in point_predicted_dict.items() if k == PredictionKeys.PREDICTION.value
                }
            predicted_df = pd.DataFrame(point_predicted_dict)

        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df


class FullBayesianTemplate(BaseTemplate):
    """ Abstract class for full Bayesian prediction
    In full prediction, the prediction occurs as a function of each parameter posterior sample,
    and the prediction results are aggregated after prediction. Prediction will
    always return the median (aka 50th percentile) along with any additional percentiles that
    are specified.
    Parameters
    ----------
    n_bootstrap_draws : int
        Number of bootstrap samples to draw from the initial MCMC or VI posterior samples.
        If -1, use the original posterior draws.
    prediction_percentiles : list
        List of integers of prediction percentiles that should be returned on prediction. To avoid reporting any
        confident intervals, pass an empty list
    """
    def __init__(self, n_bootstrap_draws=-1, prediction_percentiles=None, **kwargs):
        super().__init__(**kwargs)
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles

        # set default args
        self._prediction_percentiles = None
        self._n_bootstrap_draws = self.n_bootstrap_draws

        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles))  # unique set
        self._prediction_percentiles.sort()

        if not self.n_bootstrap_draws:
            self._n_bootstrap_draws = -1

        # init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }

        self._set_static_attributes()
        self._set_model_param_names()

    @staticmethod
    def _bootstrap(num_samples, posterior_samples, n):
        """Draw `n` number of bootstrap samples from the posterior_samples.
        Args
        ----
        n : int
            The number of bootstrap samples to draw
        """
        if n < 2:
            raise IllegalArgument("Error: Number of bootstrap draws must be at least 2")

        sample_idx = np.random.choice(range(num_samples), size=n, replace=True)
        bootstrap_samples_dict = {}
        for k, v in posterior_samples.items():
            bootstrap_samples_dict[k] = v[sample_idx]

        return bootstrap_samples_dict

    def _set_aggregate_posteriors(self):
        posterior_samples = self._posterior_samples
        mean_posteriors = {}
        median_posteriors = {}

        # for each model param, aggregate using `method`
        for param_name in self._model_param_names:
            param_ndarray = posterior_samples[param_name]

            mean_posteriors.update(
                {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
            )

            median_posteriors.update(
                {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
            )

        self._aggregate_posteriors[PredictMethod.MEAN.value] = mean_posteriors
        self._aggregate_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

    def predict(self, df, decompose=False, store_prediction_array=False, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self.get_prediction_input_meta(df)

        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(
            num_samples=self.estimator.num_sample,
            posterior_samples=self._posterior_samples,
            n=self._n_bootstrap_draws
        ) if self._n_bootstrap_draws > 1 else self._posterior_samples

        predicted_dict = self._predict(
            posterior_estimates=posterior_samples, df=df, include_error=True, **kwargs
        )

        if PredictionKeys.PREDICTION.value not in predicted_dict.keys():
            raise PredictionException("cannot find the key:'{}' from return of _predict()".format(
                PredictionKeys.PREDICTION.value))

        # reduce to prediction only if decompose is not requested
        # note that unlike other template, we can filter the keys before percentiles computation
        # hence, we can reduce with mapping on PredictionKeys
        if not decompose:
            predicted_dict = {k: v for k, v in predicted_dict.items() if k == PredictionKeys.PREDICTION.value}

        if store_prediction_array:
            self.prediction_array = predicted_dict[PredictionKeys.PREDICTION.value]
        percentiles_dict = compute_percentiles(predicted_dict, self._prediction_percentiles)
        predicted_df = pd.DataFrame(percentiles_dict)
        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df


class AggregatedPosteriorTemplate(BaseTemplate):
    """ Abstract class for full aggregated posteriors prediction
    In aggregated prediction, the parameter posterior samples are reduced using `aggregate_method`
    before performing a single prediction. This template only supports aggregated posterior inference.
    Parameters
    ----------
    aggregate_method : { 'mean', 'median' }
        Method used to reduce parameter posterior samples
    n_bootstrap_draws : int
        Number of bootstrap samples to draw from the error part to generate the uncertainty.
        If -1, will use the original posterior draw (no uncertainty).
    prediction_percentiles : list
        List of integers of prediction percentiles that should be returned on prediction. To avoid reporting any
        confident intervals, pass an empty list
    """
    def __init__(self, aggregate_method='mean', n_bootstrap_draws=1e4, prediction_percentiles=None, **kwargs):
        super().__init__(**kwargs)
        # n_bootstrap_draws here only to provide empirical prediction percentiles;
        # mid-point estimate is always replaced
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles
        self._prediction_percentiles = None
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles ))  # unique set
        self._prediction_percentiles .sort()
        # unlike full prediction, it does not take negative number of bootstrap draw
        # if self.n_bootstrap_draws < 2:
        #     raise IllegalArgument("Error: Number of bootstrap draws must be at least 2")

        self.aggregate_method = aggregate_method
        # override init aggregate posteriors
        self._aggregate_posteriors = {aggregate_method: dict()}
        self._validate_aggregate_method()

        # init aggregate posteriors
        self._aggregate_posteriors = {
            PredictMethod.MEAN.value: dict(),
            PredictMethod.MEDIAN.value: dict(),
        }

        self._set_static_attributes()
        self._set_model_param_names()

    def _validate_aggregate_method(self):
        if self.aggregate_method not in list(self._aggregate_posteriors.keys()):
            raise PredictionException("No aggregate method defined for: `{}`".format(self.aggregate_method))

    def _set_aggregate_posteriors(self):
        posterior_samples = self._posterior_samples
        mean_posteriors = {}
        median_posteriors = {}

        # for each model param, aggregate using `method`
        for param_name in self._model_param_names:
            param_ndarray = posterior_samples[param_name]

            mean_posteriors.update(
                {param_name: np.mean(param_ndarray, axis=0, keepdims=True)},
            )

            median_posteriors.update(
                {param_name: np.median(param_ndarray, axis=0, keepdims=True)},
            )

        self._aggregate_posteriors[PredictMethod.MEAN.value] = mean_posteriors
        self._aggregate_posteriors[PredictMethod.MEDIAN.value] = median_posteriors

    def fit(self, df, keep_posterior_samples=True):
        super().fit(df)
        self._set_aggregate_posteriors()
        if not keep_posterior_samples:
            self._posterior_samples = {}

    def predict(self, df, decompose=False, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")
        # obtain basic meta data from input df
        self.get_prediction_input_meta(df)

        # perform point prediction
        aggregate_posteriors = self._aggregate_posteriors.get(self.aggregate_method)
        point_predicted_dict = self._predict(
            posterior_estimates=aggregate_posteriors, df=df, include_error=False, **kwargs
        )
        for k, v in point_predicted_dict.items():
            point_predicted_dict[k] = np.squeeze(v, 0)

        # to derive confidence interval; the condition should be sufficient since we add [50] by default
        if self.n_bootstrap_draws > 0 and len(self._prediction_percentiles) > 1:
            # perform bootstrap; we don't have posterior samples. hence, we just repeat the draw here.
            posterior_samples = {}
            for k, v in aggregate_posteriors.items():
                posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)
            predicted_dict = self._predict(
                posterior_estimates=posterior_samples, df=df, include_error=True, **kwargs
            )
            percentiles_dict = compute_percentiles(predicted_dict, self._prediction_percentiles)
            # replace mid point prediction by point estimate
            percentiles_dict.update(point_predicted_dict)

            if PredictionKeys.PREDICTION.value not in percentiles_dict.keys():
                raise PredictionException("cannot find the key:'{}' from return of _predict()".format(
                    PredictionKeys.PREDICTION.value))

            # reduce to prediction only if decompose is not requested
            if not decompose:
                k = PredictionKeys.PREDICTION.value
                reduced_keys = [k + "_" + str(p) if p != 50 else k for p in self._prediction_percentiles]
                percentiles_dict = {k: v for k, v in percentiles_dict.items() if k in reduced_keys}
            predicted_df = pd.DataFrame(percentiles_dict)
        else:
            # reduce to prediction only if decompose is not requested
            if not decompose:
                point_predicted_dict = {
                    k: v for k, v in point_predicted_dict.items() if k == PredictionKeys.PREDICTION.value
                }
            predicted_df = pd.DataFrame(point_predicted_dict)

        predicted_df = prepend_date_column(predicted_df, df, self.date_col)
        return predicted_df