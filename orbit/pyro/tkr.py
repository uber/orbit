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
        """
        Notes
        -----
        A lite version of gam. No positive regressors.  Auto-seasonality.

        Labeling system:
        1. for kernel level of parameters such as rho, span, nkots, kerenel etc.,
        use suffix _lev and _coef for levels and regression to partition
        2. for knots level of parameters such as coef, loc and scale priors,
        use prefix _lev and _rr _pr for levels, regular and positive regressors to partition
        3. reduce ambigious by replacing all greeks by labels more intuitive
        use _coef, _weight etc. instead of _beta, use _scale instead of _sigma
        """
        response = self.response
        n_obs = self.n_obs
        sdy = self.sdy
        dof = self.dof

        rr = self.rr
        n_rr = self.n_rr

        k_lev = self.k_lev
        k_coef = self.k_coef
        n_knots_lev = self.n_knots_lev
        n_knots_coef = self.n_knots_coef

        lev_knot_scale = self.lev_knot_scale

        # expand dim to n_rr x n_knots_coef
        rr_knot_loc = self.rr_knot_loc.unsqueeze(-1).repeat([1, n_knots_coef])
        rr_knot_scale = self.rr_knot_scale.unsqueeze(-1).repeat([1, n_knots_coef])

        extra_out = {}

        # levels sampling
        lev_knot = pyro.sample("lev_knot", dist.Laplace(0, lev_knot_scale).expand([n_knots_lev]))
        lev = (lev_knot @ k_lev.transpose(-2, -1))

        # regular regressor (rr) sampling
        if n_rr > 0:
            coef_knot = pyro.sample("coef_knot", dist.Normal(
                rr_knot_loc,
                rr_knot_scale
                ).to_event(1)
            )
            coef = (coef_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)
        else:
            # TODO: confirm with Fritz do I need pyro.deterministric()?
            coef_knot = torch.zeros(n_knots_coef)
            coef = torch.zeros(rr.shape)

        yhat = lev + (rr * coef).sum(-1)

        # to help adjust regression in the right range
        pyro.sample("init_lev", dist.Normal(response[0], sdy), obs=lev[..., 0])

        obs_scale = pyro.sample("obs_scale", dist.HalfCauchy(sdy))
        with pyro.plate("response_plate", n_obs):
            pyro.sample("response", dist.StudentT(dof, yhat[..., :], obs_scale), obs=response)

        extra_out.update({'yhat': yhat, 'lev': lev, 'coef': coef, 'coef_knot': coef_knot})
        return extra_out
