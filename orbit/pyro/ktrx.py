import numpy as np
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.reparam import LocScaleReparam, SymmetricStableReparam

# FIXME: this is sort of dangerous; consider better implementation later
torch.set_default_tensor_type('torch.DoubleTensor')
pyro.enable_validation(True)


# class TruncatedNormal(dist.Rejector):
#     def __init__(self, loc, scale, min_x0=None):
#         propose = dist.Normal(loc, scale)
#         if min_x0 is None:
#             min_x0 = torch.zeros(1)
#
#         def log_prob_accept(x):
#             return (x > min_x0).type_as(x).log()
#
#         log_scale = torch.log(1 - dist.Normal(loc, scale).cdf(min_x0))
#         super(TruncatedNormal, self).__init__(propose, log_prob_accept, log_scale)


class Model:
    max_plate_nesting = 1

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                # TODO: this is hackk way; please fix this later
                if key in ['which_valid_res']:
                    # to use as index, tensor type has to be long or int
                    value = torch.tensor(value)
                elif key in ['coef_prior_list']:
                    pass
                else:
                    # loc/scale cannot be in long format
                    # sometimes they may be supplied as int, so dtype conversion is needed
                    value = torch.tensor(value, dtype=torch.double)
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
            min_residuals_sd = torch.tensor(1.0)
        if min_residuals_sd < 0:
            min_residuals_sd = torch.tensor(0.0)
        # expand dim to n_rr x n_knots_coef
        rr_init_knot_loc = self.rr_init_knot_loc
        rr_init_knot_scale = self.rr_init_knot_scale
        rr_knot_scale = self.rr_knot_scale

        # this does not need to expand dim since it is used as latent grand mean
        pr_init_knot_loc = self.pr_init_knot_loc
        pr_init_knot_scale = self.pr_init_knot_scale
        pr_knot_scale = self.pr_knot_scale

        # transformation of data
        regressors = torch.zeros(n_obs)
        if n_pr > 0 and n_rr > 0:
            regressors = torch.cat([rr, pr], dim=-1)
        elif n_pr > 0:
            regressors = pr
        elif n_rr > 0:
            regressors = rr

        response_tran = response - meany - seas_term

        # sampling begins here
        extra_out = {}

        # levels sampling
        lev_knot_tran = pyro.sample(
            "lev_knot_tran",
            dist.Normal(lev_knot_loc - meany, lev_knot_scale).expand([n_knots_lev]).to_event(1)
        )
        lev = (lev_knot_tran @ k_lev.transpose(-2, -1))

        # using hierarchical priors vs. multivariate priors
        if mvn == 0:
            # regular regressor sampling
            if n_rr > 0:
                # pooling latent variables
                rr_init_knot = pyro.sample(
                    "rr_init_knot", dist.Normal(
                        rr_init_knot_loc,
                        rr_init_knot_scale).to_event(1)
                )
                rr_knot = pyro.sample(
                    "rr_knot",
                    dist.Normal(
                        rr_init_knot.unsqueeze(-1) * torch.ones(n_rr, n_knots_coef),
                        rr_knot_scale).to_event(2)
                )
                rr_coef = (rr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)

            # positive regressor sampling
            if n_pr > 0:
                if geometric_walk:
                    # TODO: development method
                    pr_init_knot = pyro.sample(
                        "pr_init_knot",
                        dist.FoldedDistribution(
                            dist.Normal(pr_init_knot_loc, pr_init_knot_scale)
                        ).to_event(1)
                    )
                    pr_knot_step = pyro.sample(
                        "pr_knot_step",
                        # note that unlike rr_knot, the first one is ignored as we use the initial scale
                        # to sample the first knot
                        dist.Normal(torch.zeros(n_pr, n_knots_coef), pr_knot_scale).to_event(2)
                    )
                    pr_knot = pr_init_knot.unsqueeze(-1) * pr_knot_step.cumsum(-1).exp()
                    pr_coef = (pr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)
                else:
                    # TODO: original method
                    # pooling latent variables
                    pr_init_knot = pyro.sample(
                        "pr_knot_loc",
                        dist.FoldedDistribution(
                            dist.Normal(pr_init_knot_loc,
                                        pr_init_knot_scale)
                        ).to_event(1)
                    )

                    pr_knot = pyro.sample(
                        "pr_knot",
                        dist.FoldedDistribution(
                            dist.Normal(
                                pr_init_knot.unsqueeze(-1) * torch.ones(n_pr, n_knots_coef),
                                pr_knot_scale)
                        ).to_event(2)
                    )
                    pr_coef = (pr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)
        else:
            # regular regressor sampling
            if n_rr > 0:
                rr_init_knot = pyro.deterministic("rr_init_knot", torch.zeros(rr_init_knot_loc.shape))

                # updated mod
                loc_temp = rr_init_knot_loc.unsqueeze(-1) * torch.ones(n_rr, n_knots_coef)
                scale_temp = torch.diag_embed(rr_init_knot_scale.unsqueeze(-1) * torch.ones(n_rr, n_knots_coef))

                # the sampling
                rr_knot = pyro.sample(
                    "rr_knot",
                    dist.MultivariateNormal(
                        loc=loc_temp,
                        covariance_matrix=scale_temp
                    ).to_event(1)
                )
                rr_coef = (rr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)

            # positive regressor sampling
            if n_pr > 0:
                # this part is junk just so that the pr_init_knot has a prior; but it does not connect to anything else
                # pooling latent variables
                pr_init_knot = pyro.sample(
                    "pr_init_knot",
                    dist.FoldedDistribution(
                        dist.Normal(pr_init_knot_loc,
                                    pr_init_knot_scale)
                    ).to_event(1)
                )
                # updated mod
                loc_temp = pr_init_knot_loc.unsqueeze(-1) * torch.ones(n_pr, n_knots_coef)
                scale_temp = torch.diag_embed(pr_init_knot_scale.unsqueeze(-1) * torch.ones(n_pr, n_knots_coef))

                pr_knot = pyro.sample(
                    "pr_knot",
                    dist.MultivariateNormal(
                        loc=loc_temp,
                        covariance_matrix=scale_temp
                    ).to_event(1)
                )
                pr_knot = torch.exp(pr_knot)
                pr_coef = (pr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)

        # concatenating all latent variables
        coef_init_knot = torch.zeros(n_rr + n_pr)
        coef_knot = torch.zeros((n_rr + n_pr, n_knots_coef))

        coef = torch.zeros(n_obs)
        if n_pr > 0 and n_rr > 0:
            coef_knot = torch.cat([rr_knot, pr_knot], dim=-2)
            coef_init_knot = torch.cat([rr_init_knot, pr_init_knot], dim=-1)
            coef = torch.cat([rr_coef, pr_coef], dim=-1)
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
                m = torch.tensor(x['prior_mean'])
                sd = torch.tensor(x['prior_sd'])
                # tp = torch.tensor(x['prior_tp_idx'])
                # idx = torch.tensor(x['prior_regressor_col_idx'])
                start_tp_idx = x['prior_start_tp_idx']
                end_tp_idx = x['prior_end_tp_idx']
                idx = x['prior_regressor_col_idx']
                pyro.sample(
                    "prior_{}".format(name),
                    dist.Normal(m, sd).to_event(2),
                    obs=coef[..., start_tp_idx:end_tp_idx, idx]
                )

        # observation likelihood
        yhat = lev + (regressors * coef).sum(-1)
        obs_scale_base = pyro.sample("obs_scale_base", dist.Beta(2, 2)).unsqueeze(-1)
        # from 0.5 * sdy to sdy
        obs_scale = ((obs_scale_base * (1.0 - min_residuals_sd)) + min_residuals_sd) * sdy

        # with pyro.plate("response_plate", n_valid):
        #     pyro.sample("response",
        #                 dist.StudentT(dof, yhat[..., which_valid], obs_scale),
        #                 obs=response_tran[which_valid])

        pyro.sample("response",
                    dist.StudentT(dof, yhat[..., which_valid], obs_scale).to_event(1),
                    obs=response_tran[which_valid])

        lev_knot = lev_knot_tran + meany

        extra_out.update({
            'yhat': yhat + seas_term + meany,
            'lev': lev + meany,
            'lev_knot': lev_knot,
            'coef': coef,
            'coef_knot': coef_knot,
            'coef_init_knot': coef_init_knot,
            'obs_scale': obs_scale,
        })
        return extra_out
