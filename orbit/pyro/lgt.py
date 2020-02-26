import numpy as np
import torch

import pyro
import pyro.distributions as dist


class LGTModel:
    max_plate_nesting = 1  # max number of plates nested in model

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                value = torch.tensor(value, dtype=torch.float)
            self.__dict__[key] = value

        # transformed data
        self.is_seasonal = (self.seasonality > 1)
        self.lev_lower_bound = 0

    def __call__(self):
        response = self.response
        num_of_obs = self.num_of_obs

        # regression parameters
        lev_sm = pyro.sample("lev_sm", dist.Uniform(self.lev_sm_min, self.lev_sm_max))
        slp_sm = pyro.sample("slp_sm", dist.Uniform(self.slp_sm_min, self.slp_sm_max))

        # residual tuning parameters
        nu = pyro.sample("nu", dist.Uniform(self.min_nu, self.max_nu))

        # prior for residuals
        obs_sigma = pyro.sample("obs_sigma", dist.HalfCauchy(self.cauchy_sd))
        if self.num_of_pr == 0:
            pr = torch.zeros(num_of_obs)
        else:
            with pyro.plate("pr", self.num_of_pr):
                if self.fix_reg_coef_sd:
                    pr_sigma = self.pr_sigma_prior
                else:
                    # weak prior for sigma
                    pr_sigma = pyro.sample("pr_sigma",
                                           dist.FoldedDistribution(dist.Cauchy(
                                               self.pr_sigma_prior, self.reg_sigma_sd)))
                # weak prior for betas
                # FIXME this should be constrained to [0, self.beta_max]
                pr_beta = pyro.sample("pr_beta",
                                      dist.FoldedDistribution(
                                          dist.Normal(self.pr_beta_prior, pr_sigma)))
            pr = self.pr_mat @ pr_beta  # FIXME is this the correct matmul?

        if self.num_of_rr == 0:
            rr = torch.zeros(num_of_obs)
        else:
            with pyro.plate("rr", self.num_of_rr):
                if self.fix_reg_coef_sd:
                    rr_sigma = self.rr_sigma_prior
                else:
                    # weak prior for sigma
                    rr_sigma = pyro.sample("rr_sigma",
                                           dist.FoldedDistribution(
                                               dist.Cauchy(self.rr_sigma_prior,
                                                           self.reg_sigma_sd)))
                # weak prior for betas
                # FIXME this should be constrained to [-self.beta_min, self.beta_max]
                rr_beta = pyro.sample("rr_beta", dist.Normal(self.rr_beta_prior, rr_sigma))
            rr = self.rr_mat @ rr_beta  # FIXME is this the correct matmul?

        r = pr + rr

        # trend parameters
        # local trend proportion
        lt_coef = pyro.sample("lt_coef", dist.Uniform(self.lt_coef_min,
                                                      self.lt_coef_max))
        # global trend proportion
        gt_coef = pyro.sample("gt_coef", dist.Uniform(self.gt_coef_min,
                                                      self.gt_coef_max))
        # global trend parameter
        gt_pow = pyro.sample("gt_pow", dist.Uniform(self.gt_pow_min,
                                                    self.gt_pow_max))

        # seasonal parameters
        if self.is_seasonal:
            # seasonality smoothing parameter
            sea_sm = pyro.sample("sea_sm", dist.Uniform(self.sea_sm_min,
                                                        self.sea_sm_max))
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
            s = torch.zeros(num_of_obs)

        # states initial condition
        b[0] = torch.zeros_like(slp_sm)
        l[0] = (response[0] - r[0]).expand_as(b[0])

        for t in range(1, num_of_obs):
            # this update equation with l[t-1] ONLY.
            # intentionally different from the Holt-Winter form
            # this change is suggested from Slawek's original SLGT model
            l[t] = lev_sm * (response[t] - s[t] - r[t]) + (1 - lev_sm) * l[t - 1]
            b[t] = slp_sm * (l[t] - l[t - 1]) + (1 - slp_sm) * b[t - 1]
            if self.is_seasonal:
                s[t + self.seasonality] = \
                    sea_sm * (response[t] - l[t] - r[t]) + (1 - sea_sm) * s[t]

        # vectorize as much math as possible
        b = torch.stack(b, dim=-1).reshape(b[0].shape[:-1] + (-1,))
        l = torch.stack(l, dim=-1).reshape(l[0].shape[:-1] + (-1,))
        s = torch.stack(s, dim=-1).reshape(s[0].shape[:-1] + (-1,))

        lgt_sum = l + gt_coef * l.abs() ** gt_pow + lt_coef * b
        lgt_sum = torch.cat([l[..., :1], lgt_sum[..., :-1]], dim=-1)  # shift by 1
        yhat = lgt_sum + s[..., :num_of_obs] + r

        with pyro.plate("response_plate", num_of_obs - 1):
            pyro.sample("response", dist.StudentT(nu, yhat[..., 1:], obs_sigma),
                        obs=response[1:])

        return {'b': b, 'l': l, 's': s, 'lgt_sum': lgt_sum}


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import os
    from orbit.lgt import LGT
    from orbit.utils.plot import plot_predicted_data
    from orbit.utils.plot import plot_predicted_components

    DATA_FILE = "../../examples/data/iclaims.example.csv"
    raw_df = pd.read_csv(DATA_FILE, parse_dates=['week'])
    df = raw_df.copy()
    test_size = 52
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    lgt_map = LGT(
        response_col="claims",
        date_col="week",
        seasonality=52,
        seed=8888,
        inference_engine='stan',
        predict_method='map',
        auto_scale=False,
        is_multiplicative=True
    )
    lgt_map.fit(df=train_df)
    predicted_df = lgt_map.predict(df=test_df)