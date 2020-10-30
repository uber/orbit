import numpy as np
import torch

import pyro
import pyro.distributions as dist

torch.set_default_tensor_type('torch.DoubleTensor')

class Model:
    max_plate_nesting = 1

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                value = torch.tensor(value, dtype=torch.double)
            self.__dict__[key] = value

    def __call__(self):
        response = self.response
        pr = self.pr
        rr = self.rr
        num_pr = self.num_pr
        num_rr = self.num_rr
        t = self.t
        k_lev = self.k_lev
        # ks_lev = self.ks_lev
        k_coef = self.k_coef
        # ks_coef = self.ks_coef
        n_knots_lev = self.n_knots_lev
        n_knots_coef = self.n_knots_coef
        regressors = torch.zeros(t)
        if num_pr > 0 and num_rr > 0:
            regressors = torch.cat([rr, pr], dim=-1)
        elif num_pr > 0:
            regressors = pr
        elif num_rr > 0:
            regressors = rr

        lev_lat_sigma = self.lev_lat_sigma

        # expand dim to num_rr x n_knots_coef
        rr_beta_lat_loc = self.rr_beta_lat_loc.unsqueeze(-1).repeat([1, n_knots_coef])
        rr_beta_lat_scale = self.rr_beta_lat_scale.unsqueeze(-1).repeat([1, n_knots_coef])

        # this does not need to expand dim since it is used as latent grand mean
        pr_beta_lat_loc = self.pr_beta_lat_loc
        pr_beta_lat_scale = self.pr_beta_lat_scale
        pr_beta_step_scale = self.pr_beta_step_scale

        sdy = self.sdy
        extra_out = {}

        # levels sampling
        lev_lat = pyro.sample("lev_lat", dist.Laplace(0, lev_lat_sigma).expand([n_knots_lev]))
        lev = (lev_lat @ k_lev.transpose(-2, -1))

        # regular regressor sampling
        if num_rr > 0:
            rr_lat = pyro.sample("rr_lat", dist.Normal(
                rr_beta_lat_loc,
                rr_beta_lat_scale
                ).to_event(1)
            )
            rr_beta = (rr_lat @ k_coef.transpose(-2, -1)).transpose(-2, -1)

        # positive regressor sampling
        if num_pr > 0:
            pr_lat_mean = pyro.sample(
                "pr_lat_mean",
                dist.FoldedDistribution(
                    dist.Normal(pr_beta_lat_loc, pr_beta_lat_scale)
                )
            ).unsqueeze(-1) * torch.ones(num_pr, n_knots_coef)
            pr_lat = pyro.sample(
                "pr_lat",
                dist.FoldedDistribution(
                    dist.Normal(pr_lat_mean, pr_beta_step_scale)
                ).to_event(1)
            )
            pr_beta = (pr_lat @ k_coef.transpose(-2, -1)).transpose(-2, -1)

        # concatenating all latent variables
        beta_lat = torch.zeros(n_knots_coef)
        beta = torch.zeros(t)
        if num_pr > 0 and num_rr > 0:
            beta_lat = torch.cat([rr_lat, pr_lat], dim=-2)
            beta = torch.cat([rr_beta, pr_beta], dim=-1)
        elif num_pr > 0:
            beta_lat = pr_lat
            beta = pr_beta
        elif num_rr > 0:
            beta_lat = rr_lat
            beta = rr_beta
        yhat = lev + (regressors * beta).sum(-1)

        # inject customize priors for coef at time t
        n_prior = self.n_prior
        if n_prior > 0:
            prior_mean = self.prior_mean
            prior_sd = self.prior_sd
            prior_tp_idx = self.prior_tp_idx.int()
            prior_idx = self.prior_idx.int()

            for m, sd, tp, idx in zip(prior_mean, prior_sd, prior_tp_idx, prior_idx):
                pyro.sample("prior_{}_{}".format(tp, idx), dist.Normal(m, sd),
                            obs=beta[..., tp, idx])

        pyro.sample("init_lev", dist.Normal(response[0], sdy), obs=lev[..., 0])

        obs_sigma = pyro.sample("obs_sigma", dist.HalfCauchy(sdy))
        with pyro.plate("response_plate", t):
            pyro.sample("response", dist.StudentT(30, yhat[..., :], obs_sigma), obs=response)

        extra_out.update({'yhat': yhat, 'lev': lev, 'beta': beta, 'beta_lat': beta_lat})
        return extra_out

# class Model:
#     max_plate_nesting = 1  # max number of plates nested in model

#     def __init__(self, data):
#         for key, value in data.items():
#             key = key.lower()
#             if isinstance(value, (list, np.ndarray)):
#                 value = torch.tensor(value, dtype=torch.double)
#             self.__dict__[key] = value

#         # transformed data
#         self.is_seasonal = (self.seasonality > 1)
#         self.lev_lower_bound = 0

#     def __call__(self):
#         response = self.response
#         num_of_obs = self.num_of_obs
#         extra_out = {}

#         # smoothing params
#         if self.lev_sm_input < 0:
#             lev_sm = pyro.sample("lev_sm", dist.Uniform(0, 1))
#         else:
#             lev_sm = torch.tensor(self.lev_sm_input, dtype=torch.double)
#             extra_out['lev_sm'] = lev_sm
#         if self.slp_sm_input < 0:
#             slp_sm = pyro.sample("slp_sm", dist.Uniform(0, 1))
#         else:
#             slp_sm = torch.tensor(self.slp_sm_input, dtype=torch.double)
#             extra_out['slp_sm'] = slp_sm

#         # residual tuning parameters
#         nu = pyro.sample("nu", dist.Uniform(self.min_nu, self.max_nu))

#         # prior for residuals
#         obs_sigma = pyro.sample("obs_sigma", dist.HalfCauchy(self.cauchy_sd))
#         if self.num_of_pr == 0:
#             pr = torch.zeros(num_of_obs)
#         else:
#             with pyro.plate("pr", self.num_of_pr):
#                 # fixed scale ridge
#                 if self.reg_penalty_type == 0:
#                     pr_sigma = self.pr_sigma_prior
#                 # auto scale ridge
#                 elif self.reg_penalty_type == 2:
#                     # weak prior for sigma
#                     pr_sigma = pyro.sample("pr_sigma", dist.HalfCauchy(self.auto_ridge_scale))
#                 # case when it is not lasso
#                 if self.reg_penalty_type != 1:
#                     # weak prior for betas
#                     pr_beta = pyro.sample("pr_beta", dist.FoldedDistribution(
#                         dist.Normal(self.pr_beta_prior, pr_sigma)))
#                 else:
#                     pr_beta = pyro.sample("pr_beta",
#                                           dist.FoldedDistribution(
#                                               dist.Laplace(self.pr_beta_prior, self.lasso_scale)))
#             pr = pr_beta @ self.pr_mat.transpose(-1, -2)

#         # regression parameters
#         if self.num_of_rr == 0:
#             rr = torch.zeros(num_of_obs)
#         else:
#             with pyro.plate("rr", self.num_of_rr):
#                 # fixed scale ridge
#                 if self.reg_penalty_type == 0:
#                     rr_sigma = self.rr_sigma_prior
#                 # auto scale ridge
#                 elif self.reg_penalty_type == 2:
#                     # weak prior for sigma
#                     rr_sigma = pyro.sample("rr_sigma", dist.HalfCauchy(self.auto_ridge_scale))
#                 # case when it is not lasso
#                 if self.reg_penalty_type != 1:
#                     # weak prior for betas
#                     rr_beta = pyro.sample("rr_beta", dist.Normal(self.rr_beta_prior, rr_sigma))
#                 else:
#                     rr_beta = pyro.sample("rr_beta", dist.Laplace(self.rr_beta_prior, self.lasso_scale))
#             rr = rr_beta @ self.rr_mat.transpose(-1, -2)

#         # a hack to make sure we don't use a dimension "1" due to rr_beta and pr_beta sampling
#         r = pr + rr
#         if r.dim() > 1:
#             r = r.unsqueeze(-2)

#         # trend parameters
#         # local trend proportion
#         lt_coef = pyro.sample("lt_coef", dist.Uniform(0, 1))
#         # global trend proportion
#         gt_coef = pyro.sample("gt_coef", dist.Uniform(-0.5, 0.5))
#         # global trend parameter
#         gt_pow = pyro.sample("gt_pow", dist.Uniform(0, 1))

#         # seasonal parameters
#         if self.is_seasonal:
#             # seasonality smoothing parameter
#             if self.sea_sm_input < 0:
#                 sea_sm = pyro.sample("sea_sm", dist.Uniform(0, 1))
#             else:
#                 sea_sm = torch.tensor(self.sea_sm_input, dtype=torch.double)
#                 extra_out['sea_sm'] = sea_sm

#             # initial seasonality
#             # 33% lift is with 1 sd prob.
#             init_sea = pyro.sample("init_sea",
#                                    dist.Normal(0, 0.33)
#                                        .expand([self.seasonality])
#                                        .to_event(1))
#             init_sea = init_sea - init_sea.mean(-1, keepdim=True)

#         b = [None] * num_of_obs  # slope
#         l = [None] * num_of_obs  # level
#         if self.is_seasonal:
#             s = [None] * (self.num_of_obs + self.seasonality)
#             for t in range(self.seasonality):
#                 s[t] = init_sea[..., t]
#             s[self.seasonality] = init_sea[..., 0]
#         else:
#             s = [torch.tensor(0.)] * num_of_obs

#         # states initial condition
#         b[0] = torch.zeros_like(slp_sm)
#         if self.is_seasonal:
#             l[0] = response[0] - r[..., 0] - s[0]
#         else:
#             l[0] = response[0] - r[..., 0]

#         # update process
#         for t in range(1, num_of_obs):
#             # this update equation with l[t-1] ONLY.
#             # intentionally different from the Holt-Winter form
#             # this change is suggested from Slawek's original SLGT model
#             l[t] = lev_sm * (response[t] - s[t] - r[..., t]) + (1 - lev_sm) * l[t - 1]
#             b[t] = slp_sm * (l[t] - l[t - 1]) + (1 - slp_sm) * b[t - 1]
#             if self.is_seasonal:
#                 s[t + self.seasonality] = \
#                     sea_sm * (response[t] - l[t] - r[..., t]) + (1 - sea_sm) * s[t]

#         # evaluation process
#         # vectorize as much math as possible
#         for lst in [b, l, s]:
#             # torch.stack requires all items to have the same shape, but the
#             # initial items of our lists may not have batch_shape, so we expand.
#             lst[0] = lst[0].expand_as(lst[-1])
#         b = torch.stack(b, dim=-1).reshape(b[0].shape[:-1] + (-1,))
#         l = torch.stack(l, dim=-1).reshape(l[0].shape[:-1] + (-1,))
#         s = torch.stack(s, dim=-1).reshape(s[0].shape[:-1] + (-1,))

#         lgt_sum = l + gt_coef * l.abs() ** gt_pow + lt_coef * b
#         lgt_sum = torch.cat([l[..., :1], lgt_sum[..., :-1]], dim=-1)  # shift by 1
#         # a hack here as well to get rid of the extra "1" in r.shape
#         if r.dim() >= 2:
#             r = r.squeeze(-2)
#         yhat = lgt_sum + s[..., :num_of_obs] + r

#         with pyro.plate("response_plate", num_of_obs - 1):
#             pyro.sample("response", dist.StudentT(nu, yhat[..., 1:], obs_sigma),
#                         obs=response[1:])

#         extra_out.update({'b': b, 'l': l, 's': s, 'lgt_sum': lgt_sum})
#         return extra_out
