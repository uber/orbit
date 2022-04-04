from abc import abstractmethod
import numpy as np
import logging

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoDelta
from pyro.optim import ClippedAdam

from .base_estimator import BaseEstimator
from ..exceptions import EstimatorException
from ..utils.pyro import get_pyro_model

logger = logging.getLogger("orbit")


class PyroEstimator(BaseEstimator):
    """Abstract PyroEstimator with shared args for all PyroEstimator child classes

    Parameters
    ----------
    num_steps : int
        Number of estimator steps in optimization
    learning_rate : float
        Estimator learning rate
    learning_rate_total_decay : float
        A config re-parameterized from ``lrd`` in :class:`~pyro.optim.ClippedAdam`. For example, 0.1 means a 90%
        reduction of the final step as of original learning rate where linear decay is implied along the steps. In the
        case of 1.0, no decay is applied.  All steps will have the constant learning rate specified by `learning_rate`.
    seed : int
        Seed int
    message : int
        Print to console every `message` number of steps
    kwargs
        Additional BaseEstimator args
    Notes
    -----
        See http://docs.pyro.ai/en/stable/_modules/pyro/optim/clipped_adam.html for optimizer details
    """

    def __init__(
        self,
        num_steps=301,
        learning_rate=0.1,
        learning_rate_total_decay=1.0,
        message=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.learning_rate_total_decay = learning_rate_total_decay
        self.message = message

    @abstractmethod
    def fit(
        self, model_name, model_param_names, data_input, fitter=None, init_values=None
    ):
        raise NotImplementedError("Concrete fit() method must be implemented")


# make the name consistent across VI
class PyroEstimatorSVI(PyroEstimator):
    """Pyro Estimator for VI Sampling

    Parameters
    ----------
    num_sample : int
        Number of samples ot draw for inference, default 100
    num_particles : int
        Number of particles used in :class: `~pyro.infer.Trace_ELBO` for SVI optimization
    init_scale : float
        Parameter used in `pyro.infer.autoguide`; recommend a larger number of small dataset
    kwargs
        Additional `PyroEstimator` class args

    """

    def __init__(self, num_sample=100, num_particles=100, init_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_sample = num_sample
        self.num_particles = num_particles
        self.init_scale = init_scale

    def fit(
        self,
        model_name,
        model_param_names,
        data_input,
        sampling_temperature,
        fitter=None,
        init_values=None,
    ):
        data_input.update({"T_STAR": sampling_temperature})
        # verbose is passed through from orbit.template.base_estimator
        verbose = self.verbose
        message = self.message
        learning_rate = self.learning_rate
        learning_rate_total_decay = self.learning_rate_total_decay
        num_sample = self.num_sample
        seed = self.seed
        num_steps = self.num_steps
        if self.verbose:
            msg_template = (
                "Using SVI (Pyro) with steps: {}, samples: {}, learning rate: {},"
                " learning_rate_total_decay: {} and particles: {}."
            )
            msg = msg_template.format(
                self.num_steps,
                self.num_sample,
                self.learning_rate,
                self.learning_rate_total_decay,
                self.num_particles,
            )
            logger.info(msg)

        pyro.set_rng_seed(seed)
        if fitter is None:
            fitter = get_pyro_model(model_name)  # abstract
        model = fitter(data_input)  # concrete

        # Perform stochastic variational inference using an auto guide.
        pyro.clear_param_store()
        guide = AutoLowRankMultivariateNormal(model, init_scale=self.init_scale)
        optim = ClippedAdam(
            {"lr": learning_rate, "lrd": learning_rate_total_decay ** (1 / num_steps)}
        )
        elbo = Trace_ELBO(num_particles=self.num_particles, vectorize_particles=True)
        loss_elbo = list()
        svi = SVI(model, guide, optim, elbo)
        for step in range(num_steps):
            loss = svi.step()
            loss_elbo.append(loss)
            if verbose and step % message == 0:
                scale_rms = guide._loc_scale()[1].detach().pow(2).mean().sqrt().item()
                logger.info(
                    "step {: >4d} loss = {:0.5g}, scale = {:0.5g}".format(
                        step, loss, scale_rms
                    )
                )

        # Extract samples.
        vectorize = pyro.plate("samples", num_sample, dim=-1 - model.max_plate_nesting)
        with pyro.poutine.trace() as tr:
            samples = vectorize(guide)()
        with pyro.poutine.replay(trace=tr.trace):
            samples.update(vectorize(model)())

        # Convert from torch.Tensors to numpy.ndarrays.
        extract = {
            name: value.detach().squeeze().numpy() for name, value in samples.items()
        }

        # make sure that model param names are a subset of stan extract keys
        invalid_model_param = set(model_param_names) - set(list(extract.keys()))
        if invalid_model_param:
            raise EstimatorException(
                "Pyro model definition does not contain required parameters"
            )

        # `stan.optimizing` automatically returns all defined parameters
        # filter out unnecessary keys
        posteriors = {param: extract[param] for param in model_param_names}
        training_metrics = {"loss_elbo": np.array(loss_elbo)}
        training_metrics.update({"loglk": extract["log_prob"]})
        training_metrics.update({"sampling_temperature": sampling_temperature})

        return posteriors, training_metrics
