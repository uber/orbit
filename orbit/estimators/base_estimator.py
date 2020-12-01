from abc import ABCMeta, abstractmethod
import numpy as np


class BaseEstimator(object):
    """Base Estimator class for both Stan and Pyro Estimator

    Parameters
    ----------
    seed : int
        seed number for initial random values
    verbose : bool
        If True, output all stan diagnostics messages

    """
    def __init__(self, seed=8888, verbose=False):
        self.seed = seed
        self.verbose = verbose

        # set random state
        np.random.seed(self.seed)

    @abstractmethod
    def fit(self, model_name, model_param_names, data_input, init_values=None):
        """

        Parameters
        ----------
        model_name : str
            name of stan model
        model_param_names : list
            list of strings of model parameters names to extract
        data_input : dict
            key-value pairs of data input as required by definition in stan model
        init_values : float or np.array
            initial sampler value. If None, 'random' is used

        Returns
        -------
        OrderedDict
            key: value pairs in which key is the model parameter name
            and value is `num_sample` x posterior values

        """
        raise NotImplementedError('Concrete fit() method must be implemented')
