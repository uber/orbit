import custom_inherit as ci
from copy import copy
import numpy as np

from ..constants.constants import PredictMethod
from ..estimators.stan_estimator import StanEstimatorMCMC
from ..utils.docstring_style import merge_numpy_docs_dedup
from ..utils.predictions import prepend_date_column, aggregate_predictions
from ..exceptions import IllegalArgument, PredictionException, AbstractMethodException


ci.store["numpy_with_merge_dedup"] = merge_numpy_docs_dedup
ci.add_style("numpy_with_merge_dedup", merge_numpy_docs_dedup)


class BaseTemplate(object, metaclass=ci.DocInheritMeta(style="numpy_with_merge_dedup")):
    """Base module for model creation

    `BaseModule` will instantiate an estimator class of `estimator_type`.

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
    """
    # data labels for sampler API (stan, pyro, numpyro etc.)
    _data_input_mapper = None
    # model name (e.g. name of `*.stan` and `*.pyro` file in package)
    _model_name = None
    # supported estimators in ..estimators
    # concrete classes should overwrite this
    _supported_estimator_types = None  # set for each model

    def __init__(self, response_col='y', date_col='ds', estimator_type=StanEstimatorMCMC, **kwargs):
        self.response_col = response_col
        self.date_col = date_col
        self.estimator_type = estimator_type
        # create concrete estimator object
        self.estimator = self.estimator_type(**kwargs)

        self._model_param_names = list()
        # init posterior samples
        # `_posterior_samples` is set by `fit()`
        self._posterior_samples = dict()
        self._aggregate_posteriors = dict()
        # validator model / estimator compatibility
        self._validate_supported_estimator_type()

    def _validate_supported_estimator_type(self):
        if self.estimator_type not in self._supported_estimator_types:
            msg_template = "Model class: {} is incompatible with Estimator: {}.  Estimator Support: {}"
            model_class = type(self)
            estimator_type = self.estimator_type
            raise IllegalArgument(
                msg_template.format(model_class, estimator_type, str(self._supported_estimator_types))
            )

    def is_fitted(self):
        # if empty dict false, else true
        return bool(self._posterior_samples)

    def fit(self, **kwargs):
        raise AbstractMethodException("Abstract method.  Model should implement concrete .fit().")

    def predict(self, **kwargs):
        raise AbstractMethodException("Abstract method.  Model should implement concrete .predict().")


class MAPTemplate(BaseTemplate):
    """ Abstract class for MAP (Maximum a Posteriori) prediction

    In this module, prediction is based on Maximum a Posteriori (aka Mode) of the posterior.
    This model only supports MAP estimating `estimator_type`s
    """

    def __init__(self, n_bootstrap_draws=1e4, prediction_percentiles=None, **kwargs):
        super().__init__(**kwargs)
        # n_bootstrap_draws here only to provide empirical prediction percentiles;
        # mid-point estimate is always replaced
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles
        self._prediction_percentiles = None
        self._set_default_args()

        # override init aggregate posteriors
        self._aggregate_posteriors = {PredictMethod.MAP.value: dict()}
        self._validate_supported_estimator_type()

    def _set_default_args(self):
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles))  # unique set
        self._prediction_percentiles.sort()
        # unlike full prediction, it does not take negative number of bootstrap draw
        if self.n_bootstrap_draws < 2:
            raise IllegalArgument("Error: Number of bootstrap draws must be at least 2")

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

    def _map_predict(self, df, predict_func, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(PredictMethod.MAP.value)

        # compute inference
        posterior_samples = {}
        for k, v in aggregate_posteriors.items():
            # in_shape = v.shape[1:]
            # create and np.tile on first (batch) dimension
            posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)

        predicted_dict = predict_func(posterior_estimates=posterior_samples, df=df, include_error=True, **kwargs)
        aggregated_df = aggregate_predictions(predicted_dict, self._prediction_percentiles)
        aggregated_df = prepend_date_column(aggregated_df, df, self.date_col)

        # compute mid-point prediction
        predicted_dict = predict_func(posterior_estimates=aggregate_posteriors, df=df, include_error=False, **kwargs)

        # replacing mid-point estimation
        for k, v in predicted_dict.items():
            aggregated_df[k] = v.flatten()

        return aggregated_df


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
        If None, use the original posterior draws.
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
        self._set_default_args()

    def _set_default_args(self):
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles))  # unique set
        self._prediction_percentiles.sort()

        if not self.n_bootstrap_draws:
            self._n_bootstrap_draws = -1

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

    def _full_bayes_predict(self, df, predict_func, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")
        # if bootstrap draws, replace posterior samples with bootstrap
        posterior_samples = self._bootstrap(
            num_samples=self.estimator.num_sample,
            posterior_samples=self._posterior_samples,
            n=self._n_bootstrap_draws
        ) if self._n_bootstrap_draws > 1 else self._posterior_samples

        predicted_dict = predict_func(posterior_estimates=posterior_samples, df=df, include_error=True, **kwargs)
        aggregated_df = aggregate_predictions(predicted_dict, self._prediction_percentiles)
        aggregated_df = prepend_date_column(aggregated_df, df, self.date_col)
        return aggregated_df


class AggregatedPosteriorTemplate(BaseTemplate):
    """ Abstract class for full aggregated posteriors prediction

    In aggregated prediction, the parameter posterior samples are reduced using `aggregate_method`
    before performing a single prediction.

    Parameters
    ----------
    aggregate_method : { 'mean', 'median' }
        Method used to reduce parameter posterior samples
    """
    def __init__(self, aggregate_method='mean', n_bootstrap_draws=1e4, prediction_percentiles=None, **kwargs):
        super().__init__(**kwargs)
        # n_bootstrap_draws here only to provide empirical prediction percentiles;
        # mid-point estimate is always replaced
        self.n_bootstrap_draws = n_bootstrap_draws
        self.prediction_percentiles = prediction_percentiles
        self._prediction_percentiles = None
        self._set_default_args()

        self.aggregate_method = aggregate_method
        # override init aggregate posteriors
        self._aggregate_posteriors = {aggregate_method: dict()}
        self._validate_aggregate_method()

    def _set_default_args(self):
        if self.prediction_percentiles is None:
            self._prediction_percentiles = [5, 95]
        else:
            self._prediction_percentiles = copy(self.prediction_percentiles)

        self._prediction_percentiles += [50]  # always find median
        self._prediction_percentiles = list(set(self._prediction_percentiles ))  # unique set
        self._prediction_percentiles .sort()
        # unlike full prediction, it does not take negative number of bootstrap draw
        if self.n_bootstrap_draws < 2:
            raise IllegalArgument("Error: Number of bootstrap draws must be at least 2")

    def _validate_aggregate_method(self):
        if self.aggregate_method not in list(self._aggregate_posteriors.keys()):
            raise PredictionException("No aggregate method defined for: `{}`".format(self.aggregate_method))

    def _aggregate_predict(self, df, predict_func, **kwargs):
        # raise if model is not fitted
        if not self.is_fitted():
            raise PredictionException("Model is not fitted yet.")

        aggregate_posteriors = self._aggregate_posteriors.get(self.aggregate_method)

        # compute inference
        posterior_samples = {}
        for k, v in aggregate_posteriors.items():
            # in_shape = v.shape[1:]
            # create and np.tile on first (batch) dimension
            posterior_samples[k] = np.repeat(v, self.n_bootstrap_draws, axis=0)

        predicted_dict = predict_func(posterior_estimates=posterior_samples, df=df, include_error=True, **kwargs)
        aggregated_df = aggregate_predictions(predicted_dict, self._prediction_percentiles)
        aggregated_df = prepend_date_column(aggregated_df, df, self.date_col)

        # compute mid-point prediction
        predicted_dict = predict_func(posterior_estimates=aggregate_posteriors, df=df, include_error=False, **kwargs)

        # replacing mid-point estimation
        for k, v in predicted_dict.items():
            aggregated_df[k] = v.flatten()

        return aggregated_df

