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
        raise NotImplementedError('Concrete fit() method must be implemented')
