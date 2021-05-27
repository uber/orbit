import numpy as np
import jax.numpy as jnp
from jax import random

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist


class Model:
    max_plate_nesting = 1

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                # TODO: this is hackk way; please fix this later
                if key in ['which_valid_res']:
                    # to use as index, tensor type has to be long or int
                    value = jnp.asarray(value, dtype=int)
                elif key in ['coef_prior_list']:
                    pass
                else:
                    # loc/scale cannot be in long format
                    # sometimes they may be supplied as int, so dtype conversion is needed
                    value = jnp.asarray(value, dtype=float)
            self.__dict__[key] = value

    def __call__(self):
        """
        Notes
        -----
        Labeling system:
        1. for kernel level of parameters such as rho, span, nkots, kerenel etc.,
        use suffix _lev and _coef for levels and regression to partition
        2. for knots level of parameters such as coef, loc and scale priors,
        use prefix _lev and _rr _pr for levels, regular and positive regressors to partition
        3. reduce ambigious by replacing all greeks by labels more intuitive
        use _coef, _weight etc. instead of _beta, use _scale instead of _sigma
        """

        response = self.response
        which_valid = self.which_valid_res

        n_obs = self.n_obs
        # n_valid = self.n_valid_res
        sdy = self.sdy
        meany = self.mean_y
        dof = self.dof
        lev_knot_loc = self.lev_knot_loc
        seas_term = self.seas_term

        pr = self.pr
        rr = self.rr
        n_pr = self.n_pr
        n_rr = self.n_rr

        k_lev = self.k_lev
        k_coef = self.k_coef
        n_knots_lev = self.n_knots_lev
        n_knots_coef = self.n_knots_coef

        lev_knot_scale = self.lev_knot_scale
        # mult var norm stuff
        mvn = self.mvn
        geometric_walk = self.geometric_walk
        min_residuals_sd = self.min_residuals_sd
        if min_residuals_sd > 1.0:
            min_residuals_sd = jnp.asarray(1.0)
        if min_residuals_sd < 0:
            min_residuals_sd = jnp.asarray(0.0)
        # expand dim to n_rr x n_knots_coef
        rr_init_knot_loc = self.rr_init_knot_loc
        rr_init_knot_scale = self.rr_init_knot_scale
        rr_knot_scale = self.rr_knot_scale

        # this does not need to expand dim since it is used as latent grand mean
        pr_init_knot_loc = self.pr_init_knot_loc
        pr_init_knot_scale = self.pr_init_knot_scale
        pr_knot_scale = self.pr_knot_scale

        # transformation of data
        regressors = jnp.zeros(n_obs)
        if n_pr > 0 and n_rr > 0:
            regressors = jnp.cat([rr, pr], dim=-1)
        elif n_pr > 0:
            regressors = pr
        elif n_rr > 0:
            regressors = rr

        response_tran = response - meany - seas_term

        # sampling begins here
        extra_out = {}

        with numpyro.plate("lev_plate", n_knots_lev):
            lev_knot_tran = numpyro.sample("lev_knot_tran", dist.Normal(lev_knot_loc - meany, lev_knot_scale))
        lev = (lev_knot_tran @ k_lev.transpose())

        # regular regressor sampling
        if n_rr > 0:
            # pooling latent variables
            rr_init_knot = numpyro.sample(
                "rr_init_knot", dist.Normal(
                    rr_init_knot_loc,
                    rr_init_knot_scale)
            )
            rr_knot = numpyro.sample(
                "rr_knot",
                dist.Normal(
                    jnp.expand_dims(rr_init_knot, -1) * jnp.ones((n_rr, n_knots_coef)),
                    rr_knot_scale)
            )
            rr_coef = (rr_knot @ k_coef.transpose()).transpose()

        # positive regressor sampling
        if n_pr > 0:
            pr_init_hidden = numpyro.sample(
                "pr_init_hidden",
                dist.FoldedDistribution(
                    dist.Normal(pr_init_knot_loc, pr_init_knot_scale)
                ).to_event(1)
            )
            pr_knot_step = numpyro.sample(
                "pr_knot_step",
                # note that unlike rr_knot, the first one is ignored as we use the initial scale
                # to sample the first knot
                dist.Normal(jnp.zeros(n_pr, n_knots_coef), pr_knot_scale).to_event(2)
            )
            pr_knot = (jnp.expand_dims(pr_init_hidden.log(), -1) + pr_knot_step.cumsum(-1)).exp()
            pr_init_knot = pr_knot[..., :, 0]
            pr_coef = (pr_knot @ k_coef.transpose((-1, -2))).transpose((-1, -2))

        # concatenating all latent variables
        coef_init_knot = jnp.zeros(n_rr + n_pr)
        coef_knot = jnp.zeros((n_rr + n_pr, n_knots_coef))
        coef = jnp.zeros(n_obs)
        if n_pr > 0 and n_rr > 0:
            coef_knot = jnp.cat([rr_knot, pr_knot], dim=-2)
            coef_init_knot = jnp.cat([rr_init_knot, pr_init_knot], dim=-1)
            coef = jnp.cat([rr_coef, pr_coef], dim=-1)
        elif n_pr > 0:
            coef_knot = pr_knot
            coef_init_knot = pr_init_knot
            coef = pr_coef
        elif n_rr > 0:
            coef_knot = rr_knot
            coef_init_knot = rr_init_knot
            coef = rr_coef

        # coefficients likelihood/priors
        coef_prior_list = self.coef_prior_list
        if coef_prior_list:
            for x in coef_prior_list:
                name = x['name']
                # TODO: we can move torch conversion to init to enhance speed
                m = jnp.asarray(x['prior_mean'])
                sd = jnp.asarray(x['prior_sd'])
                # tp = jnp.asarray(x['prior_tp_idx'])
                # idx = jnp.asarray(x['prior_regressor_col_idx'])
                start_tp_idx = x['prior_start_tp_idx']
                end_tp_idx = x['prior_end_tp_idx']
                idx = x['prior_regressor_col_idx']
                numpyro.sample(
                    "prior_{}".format(name),
                    dist.Normal(m, sd).to_event(2),
                    obs=coef[..., start_tp_idx:end_tp_idx, idx]
                )

        # observation likelihood
        yhat = lev + (regressors * coef).sum(-1)
        obs_scale_base = jnp.expand_dims(numpyro.sample("obs_scale_base", dist.Beta(2, 2)), -1)
        # from 0.5 * sdy to sdy
        obs_scale = ((obs_scale_base * (1.0 - min_residuals_sd)) + min_residuals_sd) * sdy

        numpyro.sample("response",
                       dist.StudentT(dof, yhat[..., which_valid], obs_scale).to_event(1),
                       obs=response_tran[which_valid])

        # extra_out.update({
        #     'yhat': yhat + seas_term + meany,
        #     'lev': lev + meany,
        #     'lev_knot': lev_knot,
        #     'coef': coef,
        #     'coef_knot': coef_knot,
        #     'coef_init_knot': coef_init_knot,
        #     'obs_scale': obs_scale,
        # })

        yhat = numpyro.deterministic('yhat', yhat + seas_term + meany)
        lev = numpyro.deterministic('lev', lev + meany)
        lev_knot = numpyro.deterministic('lev_knot', lev_knot_tran + meany)
        coef = numpyro.deterministic('coef', coef)
        coef_init_knot = numpyro.deterministic('coef_init_knot', coef_init_knot)
        coef_knot = numpyro.deterministic('coef_knot', coef_knot)
        obs_scale = numpyro.deterministic('obs_scale', obs_scale)



