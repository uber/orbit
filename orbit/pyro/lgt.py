import numpy as np
import torch

import pyro
import pyro.distributions as dist

torch.set_default_tensor_type('torch.DoubleTensor')


class Model:
    max_plate_nesting = 1  # max number of plates nested in model

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                value = torch.tensor(value, dtype=torch.double)
            self.__dict__[key] = value

        # transformed data
        self.is_seasonal = (self.seasonality > 1)
        self.lev_lower_bound = 0

    def __call__(self):
        response = self.response
        num_of_obs = self.num_of_obs
        extra_out = {}

        # smoothing params
        if self.lev_sm_input < 0:
            lev_sm = pyro.sample("lev_sm", dist.Uniform(0, 1))
        else:
            lev_sm = torch.tensor(self.lev_sm_input, dtype=torch.double)
            extra_out['lev_sm'] = lev_sm
        if self.slp_sm_input < 0:
            slp_sm = pyro.sample("slp_sm", dist.Uniform(0, 1))
        else:
            slp_sm = torch.tensor(self.slp_sm_input, dtype=torch.double)
            extra_out['slp_sm'] = slp_sm

        # residual tuning parameters
        nu = pyro.sample("nu", dist.Uniform(self.min_nu, self.max_nu))

        # prior for residuals
        obs_sigma = pyro.sample("obs_sigma", dist.HalfCauchy(self.cauchy_sd))

        # regression parameters
        if self.num_of_pr == 0:
            pr = torch.zeros(num_of_obs)
            pr_beta = pyro.deterministic("pr_beta", torch.zeros(0))
        else:
            with pyro.plate("pr", self.num_of_pr):
                # fixed scale ridge
                if self.reg_penalty_type == 0:
                    pr_sigma = self.pr_sigma_prior
                # auto scale ridge
                elif self.reg_penalty_type == 2:
                    # weak prior for sigma
                    pr_sigma = pyro.sample("pr_sigma", dist.HalfCauchy(self.auto_ridge_scale))
                # case when it is not lasso
                if self.reg_penalty_type != 1:
                    # weak prior for betas
                    pr_beta = pyro.sample("pr_beta", dist.FoldedDistribution(
                        dist.Normal(self.pr_beta_prior, pr_sigma)))
                else:
                    pr_beta = pyro.sample("pr_beta",
                                          dist.FoldedDistribution(
                                              dist.Laplace(self.pr_beta_prior, self.lasso_scale)))
            pr = pr_beta @ self.pr_mat.transpose(-1, -2)

        if self.num_of_nr == 0:
            nr = torch.zeros(num_of_obs)
            nr_beta = pyro.deterministic("nr_beta", torch.zeros(0))
        else:
            with pyro.plate("nr", self.num_of_nr):
                # fixed scale ridge
                if self.reg_penalty_type == 0:
                    nr_sigma = self.nr_sigma_prior
                # auto scale ridge
                elif self.reg_penalty_type == 2:
                    # weak prior for sigma
                    nr_sigma = pyro.sample("nr_sigma", dist.HalfCauchy(self.auto_ridge_scale))
                # case when it is not lasso
                if self.reg_penalty_type != 1:
                    # weak prior for betas
                    nr_beta = pyro.sample("nr_beta", dist.FoldedDistribution(
                        dist.Normal(self.nr_beta_prior, nr_sigma)))
                else:
                    nr_beta = pyro.sample("nr_beta",
                                          dist.FoldedDistribution(
                                              dist.Laplace(self.nr_beta_prior, self.lasso_scale)))
            nr = nr_beta @ self.nr_mat.transpose(-1, -2)

        if self.num_of_rr == 0:
            rr = torch.zeros(num_of_obs)
            rr_beta = pyro.deterministic("rr_beta", torch.zeros(0))
        else:
            with pyro.plate("rr", self.num_of_rr):
                # fixed scale ridge
                if self.reg_penalty_type == 0:
                    rr_sigma = self.rr_sigma_prior
                # auto scale ridge
                elif self.reg_penalty_type == 2:
                    # weak prior for sigma
                    rr_sigma = pyro.sample("rr_sigma", dist.HalfCauchy(self.auto_ridge_scale))
                # case when it is not lasso
                if self.reg_penalty_type != 1:
                    # weak prior for betas
                    rr_beta = pyro.sample("rr_beta", dist.Normal(self.rr_beta_prior, rr_sigma))
                else:
                    rr_beta = pyro.sample("rr_beta", dist.Laplace(self.rr_beta_prior, self.lasso_scale))
            rr = rr_beta @ self.rr_mat.transpose(-1, -2)

        # a hack to make sure we don't use a dimension "1" due to rr_beta and pr_beta sampling
        r = pr + nr + rr
        if r.dim() > 1:
            r = r.unsqueeze(-2)

        # trend parameters
        # local trend proportion
        lt_coef = pyro.sample("lt_coef", dist.Uniform(0, 1))
        # global trend proportion
        gt_coef = pyro.sample("gt_coef", dist.Uniform(-0.5, 0.5))
        # global trend parameter
        gt_pow = pyro.sample("gt_pow", dist.Uniform(0, 1))

        # seasonal parameters
        if self.is_seasonal:
            # seasonality smoothing parameter
            if self.sea_sm_input < 0:
                sea_sm = pyro.sample("sea_sm", dist.Uniform(0, 1))
            else:
                sea_sm = torch.tensor(self.sea_sm_input, dtype=torch.double)
                extra_out['sea_sm'] = sea_sm

            # initial seasonality
            # 33% lift is with 1 sd prob.
            init_sea = pyro.sample("init_sea",
                                   dist.Normal(0, 0.33)
                                       .expand([self.seasonality])
                                       .to_event(1))
            init_sea = init_sea - init_sea.mean(-1, keepdim=True)

        b = [None] * num_of_obs  # slope
        l = [None] * num_of_obs  # level
        if self.is_seasonal:
            s = [None] * (self.num_of_obs + self.seasonality)
            for t in range(self.seasonality):
                s[t] = init_sea[..., t]
            s[self.seasonality] = init_sea[..., 0]
        else:
            s = [torch.tensor(0.)] * num_of_obs

        # states initial condition
        b[0] = torch.zeros_like(slp_sm)
        if self.is_seasonal:
            l[0] = response[0] - r[..., 0] - s[0]
        else:
            l[0] = response[0] - r[..., 0]

        # update process
        for t in range(1, num_of_obs):
            # this update equation with l[t-1] ONLY.
            # intentionally different from the Holt-Winter form
            # this change is suggested from Slawek's original SLGT model
            l[t] = lev_sm * (response[t] - s[t] - r[..., t]) + (1 - lev_sm) * l[t - 1]
            b[t] = slp_sm * (l[t] - l[t - 1]) + (1 - slp_sm) * b[t - 1]
            if self.is_seasonal:
                s[t + self.seasonality] = \
                    sea_sm * (response[t] - l[t] - r[..., t]) + (1 - sea_sm) * s[t]

        # evaluation process
        # vectorize as much math as possible
        for lst in [b, l, s]:
            # torch.stack requires all items to have the same shape, but the
            # initial items of our lists may not have batch_shape, so we expand.
            lst[0] = lst[0].expand_as(lst[-1])
        b = torch.stack(b, dim=-1).reshape(b[0].shape[:-1] + (-1,))
        l = torch.stack(l, dim=-1).reshape(l[0].shape[:-1] + (-1,))
        s = torch.stack(s, dim=-1).reshape(s[0].shape[:-1] + (-1,))

        lgt_sum = l + gt_coef * l.abs() ** gt_pow + lt_coef * b
        lgt_sum = torch.cat([l[..., :1], lgt_sum[..., :-1]], dim=-1)  # shift by 1
        # a hack here as well to get rid of the extra "1" in r.shape
        if r.dim() >= 2:
            r = r.squeeze(-2)
        yhat = lgt_sum + s[..., :num_of_obs] + r

        with pyro.plate("response_plate", num_of_obs - 1):
            pyro.sample("response", dist.StudentT(nu, yhat[..., 1:], obs_sigma),
                        obs=response[1:])

        # we care beta not the pr_beta, nr_beta, ...
        extra_out['beta'] = torch.cat([pr_beta, nr_beta, rr_beta], dim=-1)

        extra_out.update({'b': b, 'l': l, 's': s, 'lgt_sum': lgt_sum})
        return extra_out
