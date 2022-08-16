import numpy as np
import logging

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import ClippedAdam

from .base_estimator import EstimatorSVI
from ..exceptions import EstimatorException
from ..utils.pyro import get_pyro_model

logger = logging.getLogger("orbit")


# make the name consistent across VI
class PyroEstimatorSVI(EstimatorSVI):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(
            self,
            model_name,
            model_param_names,
            data_input,
            fitter=None,
            init_values=None,
            sampling_temperature=1.0,
            **kwargs,
    ):
        data_input.update({"T_STAR": sampling_temperature})
        # verbose is passed through from orbit.template.base_estimator
        verbose = self.verbose
        message = self.message
        learning_rate = self.learning_rate
        learning_rate_total_decay = self.learning_rate_total_decay
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
        vectorize = pyro.plate("samples", self.num_sample, dim=-1 - model.max_plate_nesting)
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
