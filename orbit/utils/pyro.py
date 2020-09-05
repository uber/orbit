from importlib import import_module


def get_pyro_model(model_name):
    # todo: change absolute path
    pyro_module_str = "orbit.pyro.{}".format(model_name)
    module = import_module(pyro_module_str)
    model = getattr(module, 'Model')

    return model
