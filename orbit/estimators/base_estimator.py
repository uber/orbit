from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import multiprocessing


class BaseEstimator(object):
    """Base Estimator class for both Stan and Pyro Estimator"""
    def __init__(self, verbose=False):
        self.verbose = verbose

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
