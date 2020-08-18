from abc import abstractmethod
from copy import copy

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import ClippedAdam

from .base_estimator import BaseEstimator
from ..exceptions import EstimatorException
from ..utils.general import update_dict
from ..utils.pyro import get_pyro_model


class PyroEstimator(BaseEstimator):
    """Pyro Estimator for interaction with Pyro"""
    def __init__(self, num_steps=101, learning_rate=0.1, seed=8888,
                 message=100, **kwargs):
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.seed = seed
        self.message = message

    @abstractmethod
    def fit(self, stan_model_name, model_param_names, data_input, stan_init=None):
        raise NotImplementedError('Concrete fit() method must be implemented')


class PyroEstimatorVI(PyroEstimator):
    def __init__(self, num_sample=100, pyro_vi_args=None, **kwargs):
        super().__init__(**kwargs)
        self.num_sample = num_sample
        self.pyro_vi_args = pyro_vi_args

        # init internal variable
        self._pyro_vi_args = copy(self.pyro_vi_args)

    def _set_computed_vi_args(self):
        default_pyro_vi_args = {
            'num_steps': 501,
            'learning_rate': 0.05,
        }

        self._pyro_vi_args = update_dict(default_pyro_vi_args, self._pyro_vi_args)

    def fit(self, stan_model_name, model_param_names, data_input, stan_init=None):
        verbose = self.verbose
        message = self.message
        learning_rate = self.learning_rate
        num_sample = self.num_sample
        seed = self.seed
        num_steps = self.num_steps

        pyro.set_rng_seed(seed)
        Model = get_pyro_model(stan_model_name)  # abstract
        model = Model(data_input)  # concrete

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
        vectorize = pyro.plate("samples", num_sample, dim=-1 - model.max_plate_nesting)
        with pyro.poutine.trace() as tr:
            samples = vectorize(guide)()
        with pyro.poutine.replay(trace=tr.trace):
            samples.update(vectorize(model)())

        # Convert from torch.Tensors to numpy.ndarrays.
        extract = {
            name: value.detach().squeeze().numpy()
            for name, value in samples.items()
        }

        # make sure that model param names are a subset of stan extract keys
        invalid_model_param = set(model_param_names) - set(list(extract.keys()))
        if invalid_model_param:
            raise EstimatorException("Stan model definition does not contain required parameters")

        # `stan.optimizing` automatically returns all defined parameters
        # filter out unecessary keys
        extract = {param: extract[param] for param in model_param_names}

        return extract
