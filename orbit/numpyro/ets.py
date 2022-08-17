import jax.numpy as jnp
from jax import random
import numpy as np


import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan


class Model:
    max_plate_nesting = 1

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                value = jnp.array(value)
            self.__dict__[key] = value

    def __call__(self):
        response = self.response
        seasonality = self.seasonality
        seasonality_sd = self.seasonality_sd
        response_sd = self.response_sd

        n_obs = len(response)
        lev_sm = numpyro.sample("lev_sm", dist.Uniform(0, 1))
        sea_sm = numpyro.sample("sea_sm", dist.Uniform(0, 1))
        obs_sigma = numpyro.sample("obs_sigma", dist.Cauchy(0, response_sd))

        with numpyro.plate("seasonality", seasonality - 1):
            init_sea_param = numpyro.sample("init_sea", dist.Normal(0, seasonality_sd))
        init_sea = jnp.hstack(
            (init_sea_param[1:], -1 * jnp.sum(init_sea_param), init_sea_param[0])
        )

        init_lev = jnp.reshape(response[0] - init_sea_param[0], (1, ))

        def transition_fn(carry, t):
            prev_lev, prev_sea = carry
            # forecast
            yhat = prev_lev + prev_sea[0]

            # update
            new_lev = lev_sm * (response[t] - prev_sea[0]) + (1 - lev_sm) * prev_lev
            sea_update = sea_sm * (response[t] - new_lev) + (1 - sea_sm) * prev_sea[0]
            new_sea = jnp.concatenate((prev_sea[1:], sea_update), 0)

            return (new_lev, new_sea), (new_lev, sea_update, yhat)

        _, res = scan(
            transition_fn, (init_lev, init_sea), jnp.arange(1, n_obs)
        )
        l_generated, s_generated, yhat_generated = res
        # the last dimension is an extra one; collaspe them
        l_generated = jnp.squeeze(l_generated, -1)
        s_generated = jnp.squeeze(s_generated, -1)
        yhat_generated = jnp.squeeze(yhat_generated, -1)

        numpyro.sample("response", dist.Normal(yhat_generated, obs_sigma), obs=response[1:])

        s = numpyro.deterministic("s", jnp.concatenate((init_sea_param[0, None], init_sea, s_generated), 0))
        l = numpyro.deterministic("l", jnp.concatenate((init_lev, l_generated), -1))
        yhat = numpyro.deterministic("yhat", jnp.concatenate((response[0, None], yhat_generated), 0))
        extra_out = {
            "s": s,
            "l": l,
            "yhat": yhat,
        }
        return extra_out
