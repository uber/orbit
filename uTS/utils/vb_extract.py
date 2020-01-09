import numpy as np
from collections import OrderedDict


def vb_extract(results):
    """Re-arrange and extract posteriors from variational inference fit from stan

    Due to different structure of the output from fit from vb, we need this additional logic to
    extract posteriors.  The logic is based on
    https://gist.github.com/lwiklendt/9c7099288f85b59edc903a5aed2d2d64

    Args
    ----
    results: dict
        dict exported from pystan.StanModel object by `vb` method

    Returns
    -------
    params: OrderedDict
        dict of arrays where each element represent arrays of samples (Index of Sample, Sample
        dimension 1, Sample dimension 2, ...)
    """
    param_specs = results['sampler_param_names']
    samples = results['sampler_params']
    n = len(samples[0])

    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
            idxs = [int(i) for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # create arrays
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape))) for name, shape in param_shapes.items()])

    # second pass, set arrays
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # -1 because pystan returns 1-based indexes for vb!
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params

