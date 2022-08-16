from importlib import import_module


def get_pyro_model(model_name, is_num_pyro=False):
    """ Generate path and pick which module and module dir to import

    Parameters
    ----------
    model_name : str
    is_num_pyro : bool

    Returns
    -------
    numpyro.Model / pyro.Model
    """
    # todo: change absolute path
    #
    if is_num_pyro:
        module_str = "orbit.numpyro.{}".format(model_name)
        module = import_module(module_str)
        model = getattr(module, 'Model')
    else:
        module_str = "orbit.pyro.{}".format(model_name)
        module = import_module(module_str)
        model = getattr(module, 'Model')

    return model
