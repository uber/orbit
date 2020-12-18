import numpy as np
import torch

import pyro
import pyro.distributions as dist

# FIXME: this is sort of dangerous; consider better implementation later
torch.set_default_tensor_type('torch.DoubleTensor')

pyro.enable_validation(True)


class Model:
    max_plate_nesting = 1

    def __init__(self, data):
        for key, value in data.items():
            key = key.lower()
            if isinstance(value, (list, np.ndarray)):
                if key in ['which_valid_res']:
                    # to use as index, tensor type has to be long or int
                    value = torch.tensor(value)
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
        n_obs = self.n_obs
        n_valid = self.n_valid_res
        sdy = self.sdy
        meany = self.mean_y
        dof = self.dof

        response = self.response
        which_valid = self.which_valid_res

        n_knots_lev = self.n_knots_lev
        lev_knot_scale = self.lev_knot_scale
        k_lev = self.k_lev

        p = self.p
        # expand dim to n_rr x n_knots_coef
        coef_knot_pool_loc = self.coef_knot_pool_loc
        coef_knot_pool_scale = self.coef_knot_pool_scale
        coef_knot_scale = self.coef_knot_scale.unsqueeze(-1)
        regressors = self.regressors
        k_coef = self.k_coef
        n_knots_coef = self.n_knots_coef

        response_tran = response - meany

        # sampling begins here
        extra_out = {}

        # levels sampling
        lev_drift = pyro.sample("lev_drift", dist.Laplace(0, lev_knot_scale).expand([n_knots_lev]).to_event(1))
        lev_knot_tran = lev_drift.cumsum(-1)
        lev_tran = (lev_knot_tran @ k_lev.transpose(-2, -1))

        # regular regressor sampling
        regression = torch.zeros(lev_tran.shape)
        if p > 0:
            # pooling latent variables
            coef_knot_loc = pyro.sample(
                "coef_knot_loc", dist.Normal(
                    coef_knot_pool_loc,
                    coef_knot_pool_scale).to_event(1)
            )
            coef_knot = pyro.sample(
                "coef_knot",
                dist.Normal(
                    coef_knot_loc.unsqueeze(-1) * torch.ones(p, n_knots_coef),
                    coef_knot_scale).to_event(2)
            )
            coef = (coef_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)
            regression = (regressors * coef).sum(-1)

        # concatenating all latent variables
        yhat = lev_tran + regression

        obs_scale = pyro.sample("obs_scale", dist.HalfCauchy(sdy)).unsqueeze(-1)
        # with pyro.plate("response_plate", n_valid):
        pyro.sample("response",
                    dist.StudentT(dof, yhat[..., which_valid], obs_scale).to_event(1),
                    obs=response_tran[which_valid])

        extra_out.update({
            'yhat': yhat + meany,
            'lev': lev_tran + meany,
            'lev_knot': lev_knot_tran + meany,
            'coef': coef,
        })
        return extra_out
