from __future__ import absolute_import, division, print_function
from pystan import StanModel
import pickle
import pkg_resources
import os
import numpy as np


def compile_stan_model(stan_model_name):
    """
    Compile stan model and save as pkl
    """
    source_model = pkg_resources.resource_filename(
        'orbit',
        'stan/{}.stan'.format(stan_model_name)
    )
    compiled_model = pkg_resources.resource_filename(
        'orbit',
        'stan_compiled/{}.pkl'.format(stan_model_name)
    )

    # updated for py3
    os.makedirs(os.path.dirname(compiled_model), exist_ok=True)

    # compile if stan source has changed
    if not os.path.isfile(compiled_model) or \
            os.path.getmtime(compiled_model) < os.path.getmtime(source_model):

        with open(source_model) as f:
            model_code = f.read()

        sm = StanModel(model_code=model_code)

        with open(compiled_model, 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def get_compiled_stan_model(stan_model_name):
    """
    Load compiled Stan model
    """

    compile_stan_model(stan_model_name)

    model_file = pkg_resources.resource_filename(
        'orbit',
        'stan_compiled/{}.pkl'.format(stan_model_name)
    )
    with open(model_file, 'rb') as f:
        return pickle.load(f)


def estimate_level_smoothing(x, seasonality=1, horizon=None):
    """ Improving estimation of level smoothing by running a simple smoothing on differenced data
        Parameters
    ----------
    x: 1-D array-like
        Input of observations
    horizon: int
        Forecast horizon used to test robustness of estimation
    seasonality: int
        The highest seasonal order describing the
        input observations
    Returns
    -------
    float:
        estimated level smoothing parameters
    """
    compiled_stan_model = get_compiled_stan_model('simple_smoothing')
    if not horizon:
        horizon = seasonality
    data = {
        'N_OBS': len(x),
        'RESPONSE': x,
        'HORIZON': horizon,
        'SDY': np.std(x),
        'SEASONALITY': seasonality,
    }
    op = compiled_stan_model.optimizing(data)
    return op['lev_sm']