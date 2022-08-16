from abc import abstractmethod
import numpy as np
from ..utils.general import update_dict


# from ..utils.docstring_style import merge_numpy_docs_dedup
# import custom_inherit as ci
# ci.store["numpy_with_merge_dedup"] = merge_numpy_docs_dedup
# ci.add_style("numpy_with_merge_dedup", merge_numpy_docs_dedup)


class BaseEstimator:
    """Base abstract estimator class for future estimator extension

    Parameters
    ----------
    seed : int
        seed number for initial random values
    verbose : bool
        If True (default), output all diagnostics messages from estimators

    """

    def __init__(self, seed=8888, verbose=True, **kwargs, ):
        self.seed = seed
        self.verbose = verbose

        # set random state
        np.random.seed(self.seed)

    @abstractmethod
    def fit(
            self,
            model_name,
            model_param_names,
            data_input,
            fitter=None,
            init_values=None,
            **kwargs,
    ):
        """

        Parameters
        ----------
        model_name : str
            name of model - used in mapping the right sampling file (stan/pyro/...)
        model_param_names : List
            strings of model parameters names to extract
        data_input : dict
            key-value pairs of data input as required by definition in samplers (stan/pyro/...)
        fitter :
            model object used for fitting; this will be used instead of model_name if supplied to search for
            model object
        init_values : float or np.array
            initial sampler value. If None, 'random' is used

        Returns
        -------
        posteriors : dict
            key value pairs where key is the model parameter name and value is `num_sample` x posterior values
        training_metrics : dict
            metrics and metadata related to the training process
        """
        raise NotImplementedError("Concrete fit() method must be implemented")


class EstimatorSVI(BaseEstimator):
    """Abstract PyroEstimator with shared args for all PyroEstimator child classes

    Parameters
    ----------
    num_steps : int
        Number of estimator steps in optimization
    learning_rate : float
        Estimator learning rate
    learning_rate_total_decay : float
        A config re-parameterized from ``lrd`` in :class:`~pyro.optim.ClippedAdam`. For example, 0.1 means a 90%
        reduction of the final step as of original learning rate where linear decay is implied along the steps. In the
        case of 1.0, no decay is applied.  All steps will have the constant learning rate specified by `learning_rate`.
    num_sample : int
        Number of samples ot draw for inference, default 100
    num_particles : int
        Number of particles used in :class: `~pyro.infer.Trace_ELBO` for SVI optimization
    init_scale : float
        Parameter used in `pyro.infer.autoguide`; recommend a larger number of small dataset
    seed : int
        Seed int
    message : int
        Print to console every `message` number of steps
    kwargs
        Additional BaseEstimator args
    Notes
    -----
        See http://docs.pyro.ai/en/stable/_modules/pyro/optim/clipped_adam.html for optimizer details
    """

    def __init__(
            self,
            num_steps=301,
            learning_rate=0.1,
            learning_rate_total_decay=1.0,
            message=100,
            num_sample=100,
            num_particles=100,
            init_scale=0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.learning_rate_total_decay = learning_rate_total_decay
        self.message = message
        self.num_sample = num_sample
        self.num_particles = num_particles
        self.init_scale = init_scale

    @abstractmethod
    def fit(
            self,
            model_name,
            model_param_names,
            data_input,
            fitter=None,
            init_values=None,
            sampling_temperature=1.0,
            **kwargs,
    ):
        raise NotImplementedError("Concrete fit() method must be implemented")


class EstimatorMCMC(BaseEstimator):
    """Abstract StanEstimator with shared args for all StanEstimator child classes

     Parameters
     ----------
     num_warmup : int
         Number of samples to warm up and to be discarded, default 900
     num_sample : int
         Number of samples to return, default 100
     chains : int
         Number of chains in stan sampler, default 4
     cores : int
         Number of cores for parallel processing, default max(cores, multiprocessing.cpu_count())
     kwargs
         Additional `BaseEstimator` class args

     """

    def __init__(
            self,
            num_warmup=900,
            num_sample=100,
            chains=4,
            cores=8,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_warmup = num_warmup
        self.num_sample = num_sample
        self.chains = chains
        self.cores = cores

        # init computed configs
        self._num_warmup_per_chain = None
        self._num_sample_per_chain = None
        self._num_iter_per_chain = None
        self._total_iter = None

        self._set_computed_configs()

    @abstractmethod
    def fit(
            self,
            model_name,
            model_param_names,
            data_input,
            fitter=None,
            init_values=None,
            sampling_temperature=1.0,
            **kwargs,
    ):
        raise NotImplementedError("Concrete fit() method must be implemented")

    def _set_computed_configs(self):
        """Sets sampler configs based on init class attributes"""
        raise NotImplementedError("Concrete _set_computed_configs() method must be implemented")


class EstimatorMAP(BaseEstimator):
    """Stan Estimator for MAP Posteriors

    Parameters
    ----------
    stan_map_args : dict
        Supplemental stan vi args to pass to PyStan.optimizing()

    """

    def __init__(self, algorithm="LBFGS", **kwargs):
        super().__init__(**kwargs)
        if self.algorithm is None:
            self.algorithm = "LBFGS"
        else:
            self.algorithm = algorithm

    def fit(
            self,
            model_name,
            model_param_names,
            data_input,
            fitter=None,
            init_values=None,
            **kwargs,
    ):
        raise NotImplementedError("Concrete fit() method must be implemented")

