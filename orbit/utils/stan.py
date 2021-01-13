from __future__ import absolute_import, division, print_function
from pystan import StanModel
import pickle
import pkg_resources
import os


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

        with open(source_model, encoding="utf-8") as f:
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
