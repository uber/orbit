import numpy as np
import torch

import pyro
import pyro.distributions as dist


# FIXME: this is sort of dangerous; consider better implementation later
torch.set_default_tensor_type("torch.DoubleTensor")
pyro.enable_validation(True)


class Model:
    max_plate_nesting = 1

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                # TODO: this is hackk way; please fix this later
                if key in ["which_valid_res"]:
                    # to use as index, tensor type has to be long or int
                    value = torch.tensor(value)
                elif key in ["coef_prior_list"]:
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

        n_obs = self.num_of_obs
        # n_valid = self.n_valid_res
        sdy = self.response_sd
        meany = self.mean_y
        dof = self.dof
        lev_knot_loc = self.lev_knot_loc
        seas_term = self.seas_term
        # added for tempured sampling
        T = self.t_star

        pr = self.pr
        nr = self.nr
        rr = self.rr
        n_pr = self.n_pr
        n_rr = self.n_rr
        n_nr = self.n_nr

        k_lev = self.k_lev
        k_coef = self.k_coef
        n_knots_lev = self.n_knots_lev
        n_knots_coef = self.n_knots_coef

        lev_knot_scale = self.lev_knot_scale

        resid_scale_ub = self.resid_scale_ub
        if resid_scale_ub > sdy:
            resid_scale_ub = sdy

        # expand dim to n_rr x n_knots_coef
        rr_init_knot_loc = self.rr_init_knot_loc
        rr_init_knot_scale = self.rr_init_knot_scale
        rr_knot_scale = self.rr_knot_scale

        # this does not need to expand dim since it is used as latent grand mean
        pr_init_knot_loc = self.pr_init_knot_loc
        pr_init_knot_scale = self.pr_init_knot_scale
        pr_knot_scale = self.pr_knot_scale
        nr_init_knot_loc = self.nr_init_knot_loc
        nr_init_knot_scale = self.nr_init_knot_scale
        nr_knot_scale = self.nr_knot_scale

        # prepare regressor matrix
        if n_pr == 0:
            pr = torch.zeros(0)
        if n_nr == 0:
            nr = torch.zeros(0)
        if n_rr == 0:
            rr = torch.zeros(0)
        regressors = torch.cat([rr, pr, nr], dim=-1)
        if n_pr == 0 and n_nr == 0 and n_rr == 0:
            regressors = torch.zeros(n_obs)

        response_tran = response - meany - seas_term

        # sampling begins here
        extra_out = {}

        # levels sampling
        lev_knot_tran = pyro.sample(
            "lev_knot_tran",
            dist.Normal(lev_knot_loc - meany, lev_knot_scale)
            .expand([n_knots_lev])
            .to_event(1),
        )
        lev = lev_knot_tran @ k_lev.transpose(-2, -1)
        # regular regressor sampling
        if n_rr > 0:
            # pooling latent variables
            rr_init_knot = pyro.sample(
                "rr_init_knot",
                dist.Normal(rr_init_knot_loc, rr_init_knot_scale).to_event(1),
            )
            rr_knot = pyro.sample(
                "rr_knot",
                dist.Normal(
                    rr_init_knot.unsqueeze(-1) * torch.ones(n_rr, n_knots_coef),
                    rr_knot_scale,
                ).to_event(2),
            )
            rr_coef = (rr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)

        # positive regressor sampling
        if n_pr > 0:
            # pooling latent variables
            pr_init_knot = pyro.sample(
                "pr_knot_loc",
                dist.FoldedDistribution(
                    dist.Normal(pr_init_knot_loc, pr_init_knot_scale)
                ).to_event(1),
            )

            pr_knot = pyro.sample(
                "pr_knot",
                dist.FoldedDistribution(
                    dist.Normal(
                        pr_init_knot.unsqueeze(-1) * torch.ones(n_pr, n_knots_coef),
                        pr_knot_scale,
                    )
                ).to_event(2),
            )
            pr_coef = (pr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)

        # negative regressor sampling
        if n_nr > 0:
            # pooling latent variables
            nr_init_knot = -1.0 * pyro.sample(
                "nr_knot_loc",
                dist.FoldedDistribution(
                    dist.Normal(nr_init_knot_loc, nr_init_knot_scale)
                ).to_event(1),
            )

            nr_knot = -1.0 * pyro.sample(
                "nr_knot",
                dist.FoldedDistribution(
                    dist.Normal(
                        nr_init_knot.unsqueeze(-1) * torch.ones(n_nr, n_knots_coef),
                        nr_knot_scale,
                    )
                ).to_event(2),
            )
            nr_coef = (nr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)

        if n_pr == 0:
            pr_init_knot = torch.zeros(0)
            pr_knot = torch.zeros(0)
            pr_coef = torch.zeros(0)
        if n_nr == 0:
            nr_init_knot = torch.zeros(0)
            nr_knot = torch.zeros(0)
            nr_coef = torch.zeros(0)
        if n_rr == 0:
            rr_init_knot = torch.zeros(0)
            rr_knot = torch.zeros(0)
            rr_coef = torch.zeros(0)
        coef_init_knot = torch.cat([rr_init_knot, pr_init_knot, nr_init_knot], dim=-1)
        coef_knot = torch.cat([rr_knot, pr_knot, nr_knot], dim=-2)
        coef = torch.cat([rr_coef, pr_coef, nr_coef], dim=-1)
        if n_pr == 0 and n_nr == 0 and n_rr == 0:
            # coef_init_knot = torch.zeros(n_rr + n_pr + n_nr)
            # coef_knot = torch.zeros((n_rr + n_pr + n_nr, n_knots_coef))
            coef = torch.zeros(n_obs)

        # coefficients likelihood/priors
        coef_prior_list = self.coef_prior_list
        if coef_prior_list:
            for x in coef_prior_list:
                name = x["name"]
                # TODO: we can move torch conversion to init to enhance speed
                m = torch.tensor(x["prior_mean"])
                sd = torch.tensor(x["prior_sd"])
                # tp = torch.tensor(x['prior_tp_idx'])
                # idx = torch.tensor(x['prior_regressor_col_idx'])
                start_tp_idx = x["prior_start_tp_idx"]
                end_tp_idx = x["prior_end_tp_idx"]
                idx = x["prior_regressor_col_idx"]
                pyro.sample(
                    "prior_{}".format(name),
                    dist.Normal(m, sd).to_event(2),
                    obs=coef[..., start_tp_idx:end_tp_idx, idx],
                )

        # observation likelihood
        yhat = lev + (regressors * coef).sum(-1)
        # set lower and upper bound of scale parameter
        # Beta(5, 1) set up some gravity to ask for extra evidence to reduce the scale sharply
        obs_scale_base = pyro.sample("obs_scale_base", dist.Beta(5, 1)).unsqueeze(-1)
        obs_scale = obs_scale_base * resid_scale_ub

        # this line addes a tempurature to the obs fit
        with pyro.poutine.scale(scale=1.0 / T):
            pyro.sample(
                "response",
                dist.StudentT(dof, yhat[..., which_valid], obs_scale).to_event(1),
                obs=response_tran[which_valid],
            )

        log_prob = dist.StudentT(dof, yhat[..., which_valid], obs_scale).log_prob(
            response_tran[which_valid]
        )

        lev_knot = lev_knot_tran + meany

        extra_out.update(
            {
                "yhat": yhat + seas_term + meany,
                "lev": lev + meany,
                "lev_knot": lev_knot,
                "coef": coef,
                "coef_knot": coef_knot,
                "coef_init_knot": coef_init_knot,
                "obs_scale": obs_scale,
                "log_prob": log_prob,
            }
        )
        return extra_out
