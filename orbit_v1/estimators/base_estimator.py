from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import multiprocessing


class BaseEstimator(object):
    """Base Estimator class for shared methods for Inference Engines"""
    def __init__(self, verbose=False):
        self.verbose = verbose
