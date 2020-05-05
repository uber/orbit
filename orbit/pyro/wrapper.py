from importlib import import_module


def _load_model(model_name):
    module_name, class_name = model_name.rsplit('.', 1)
    module = import_module(module_name)
    Model = getattr(module, class_name)
    return Model


def pyro_map(model_name, data, seed, num_steps=101, learning_rate=0.1, verbose=True, message=100):
    """
    Parameters
    ----------
        model_name: str
            name for pyro to search for the .py script and model
        data: dict
            all elements of data needed for sampling
        seed: int
        num_steps: int
            steps of training
        learning_rate: float
        verbose: bool
        message: int
            number of steps to retrieve information of training
    Returns
    -------
    params: OrderedDict:
        dict of all required samples
    """
    # import these lazily to avoid adding dependencies
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDelta
    from pyro.optim import ClippedAdam

    pyro.set_rng_seed(seed)
    Model = _load_model(model_name)
    model = Model(data)

    # Perform MAP inference using an AutoDelta guide.
    pyro.clear_param_store()
    guide = AutoDelta(model)
    optim = ClippedAdam({"lr": learning_rate, "betas": (0.5, 0.8)})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optim, elbo)
    for step in range(num_steps):
        loss = svi.step()
        if verbose and step % message == 0:
            print("step {: >4d} loss = {:0.5g}".format(step, loss))

    # Extract point estimates.
    values = guide()
    values.update(pyro.poutine.condition(model, values)())

    # Convert from torch.Tensors to numpy.ndarrays.
    extract = {
        name: value.detach().numpy()
        for name, value in values.items()
    }

    return extract


def pyro_svi(model_name, data, seed, num_steps=101, learning_rate=0.1, num_samples=100, verbose=True, message=100):
    """
    Parameters
    ----------
        model_name: str
            name for pyro to search for the .py script and model
        data: dict
            all elements of data needed for sampling
        seed: int
        num_steps: int
            steps of training
        learning_rate: float
        num_samples: int
        verbose: bool
        message: int
            number of steps to retrieve information of training
    Returns
    -------
    params: OrderedDict:
        dict of all required samples
    """
    # import these lazily to avoid adding dependencies
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoLowRankMultivariateNormal
    from pyro.optim import ClippedAdam

    pyro.set_rng_seed(seed)
    Model = _load_model(model_name)
    model = Model(data)

    # Perform stochastic variational inference using an auto guide.
    pyro.clear_param_store()
    guide = AutoLowRankMultivariateNormal(model)
    optim = ClippedAdam({"lr": learning_rate})
    elbo = Trace_ELBO(num_particles=100, vectorize_particles=True)
    svi = SVI(model, guide, optim, elbo)
    for step in range(num_steps):
        loss = svi.step()
        if verbose and step % message == 0:
            scale_rms = guide._loc_scale()[1].detach().pow(2).mean().sqrt().item()
            print("step {: >4d} loss = {:0.5g}, scale = {:0.5g}".format(step, loss, scale_rms))

    # Extract samples.
    vectorize = pyro.plate("samples", num_samples, dim=-1 - model.max_plate_nesting)
    with pyro.poutine.trace() as tr:
        samples = vectorize(guide)()
    with pyro.poutine.replay(trace=tr.trace):
        samples.update(vectorize(model)())

    # Convert from torch.Tensors to numpy.ndarrays.
    extract = {
        name: value.detach().squeeze().numpy()
        for name, value in samples.items()
    }

    return extract
