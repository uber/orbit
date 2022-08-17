import numpyro
from numpyro.infer import MCMC, NUTS, Predictive, autoguide, SVI, Trace_ELBO
from jax import random, local_device_count
import numpy as np
import logging

from .base_estimator import EstimatorMCMC, EstimatorMAP
from ..utils.pyro import get_pyro_model

logger = logging.getLogger("orbit")


class NumPyroEstimatorMCMC(EstimatorMCMC):
    """NumPyro Estimator for MCMC (No-U-Turn) Sampling

    Parameters
    ----------
    num_sample : int
        Number of estimator steps in optimization
    num_warmup : int
        Estimator learning rate
    seed int
    message :  int
        Print to console every `message` number of steps
    kwargs
        Additional BaseEstimator args
    """

    def _set_computed_configs(self):
        # make sure cores can only be as large as the device support
        self.cores = min(self.cores, local_device_count())
        self._num_warmup_per_chain = int(self.num_warmup / self.chains)
        self._num_sample_per_chain = int(self.num_sample / self.chains)
        self._num_iter_per_chain = (
                self._num_warmup_per_chain + self._num_sample_per_chain
        )
        self._total_iter = self._num_iter_per_chain * self.chains

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
        # TODO: some log here
        # TODO: log include message we don't use init values
        # TODO: and sampling temperature
        # fitter as abstract
        if fitter is None:
            fitter = get_pyro_model(model_name, is_num_pyro=True)

        numpyro.set_host_device_count(self.cores)
        if self.verbose:
            msg_template = (
                "Sampling (NumPyro) with chains: {:d}, cores: {:d}, "
                "warmups (per chain): {:d} and samples(per chain): {:d}."
            )
            msg = msg_template.format(
                self.chains,
                self.cores,
                self._num_warmup_per_chain,
                self._num_sample_per_chain,
            )
            logger.info(msg)

        # fitter is a class constructor
        # model is a concrete callable object
        model = fitter(data_input)
        nuts_kernel = NUTS(model)
        mcmc = MCMC(
            nuts_kernel,
            progress_bar=True,
            num_samples=self._num_sample_per_chain,
            num_warmup=self._num_warmup_per_chain,
            num_chains=self.chains,
            chain_method="parallel",
        )
        rng_key = random.PRNGKey(self.seed)
        mcmc.run(rng_key)
        extract = mcmc.get_samples()
        # convert back to numpy
        posteriors = {param: np.array(extract[param]) for param in model_param_names}

        # TODO: add more fields
        training_metrics = {}

        return posteriors, training_metrics


class NumPyroEstimatorMAP(EstimatorMAP):
    """NumPyro Estimator for Max a Posteriori(MAP)

    Parameters
    ----------

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
            **kwargs,
    ):
        if fitter is None:
            fitter = get_pyro_model(model_name, is_num_pyro=True)

        # fitter is a class constructor
        # model is a concrete callable object
        model = fitter(data_input)
        guide = autoguide.AutoDelta(
            model,
            # init_loc_fn=numpyro.infer.initialization.init_to_uniform,
        )
        optimizer = numpyro.optim.Adam(0.001)
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_results = svi.run(
            random.PRNGKey(self.seed),
            self.num_iters,
        )
        # extract latent variable point estimates
        map_extracts = guide()
        map_extracts.update(numpyro.handlers.condition(model, map_extracts)())
        posteriors = dict()
        # extract = svi_results.params
        # convert back to numpy
        posteriors = {name: np.array(value) for name, value in map_extracts.items()}

        training_metrics = dict()
        return posteriors, training_metrics
