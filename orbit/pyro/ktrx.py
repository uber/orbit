import numpy as np
import torch

import pyro
import pyro.distributions as dist

# FIXME: this is sort of dangerous; consider better implementation later
torch.set_default_tensor_type('torch.DoubleTensor')
pyro.enable_validation(True)

torch.set_printoptions(precision=16, threshold = 100000, linewidth  = 1000)
                       
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
        
        # add in the rho_coef 
        rho_coef = self.rho_coefficients
        est_rho = self.est_rho
        knots_tp =  self._knots_tp_coefficients

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

        # sample coef rho
        
        
        rho_coef_from_mod = 2*rho_coef
        # hack for saving results 
        if not est_rho:
            rho_coef = pyro.sample('rho_coef',
                                   dist.Uniform(low = 0.00, high = 1.00))
        
        if est_rho:
            k_coef2 = self.k_coef 
            rho_coef = pyro.sample('rho_coef',
                                   dist.Uniform(low = 0.0, high =  rho_coef_from_mod))
            
            rho_sq_t2 = 2*(rho_coef**2)

            tp = np.arange(1, n_obs + 1) / n_obs
            # make a x = nxm where each column is tp 
            x = torch.tensor(tp).unsqueeze(1)
            x = x.repeat(1, len(knots_tp))
            # max x_i = nxm where each row is x_i 
            x_i = torch.tensor(knots_tp).unsqueeze(1)
            x_i = x_i.repeat(1, len(tp))
            x_i = x_i.transpose(0,-1)
            k_temp = torch.pow((x - x_i),2) 
            if len(rho_sq_t2.size()) >= 1:
                #rho_sq_t2 = rho_sq_t2*0 + 2*(0.05**2) # for debugging only 
                k_temp = k_temp.repeat(rho_sq_t2.size()[0],1,1) # this line must be fixed 
                rho_sq_t2 = rho_sq_t2.view(-1, 1, 1)    
                k_temp = k_temp / rho_sq_t2
                k_temp[k_temp >= 400.0] = 400.0 
                k_coef = torch.exp(-1 * k_temp )
                # add in the nomrialize 
                k_norm = torch.sum(k_coef, -1, keepdim=True)
                #k_norm[k_norm<=0.0001] = 0.0001
                k_coef = torch.divide(k_coef,k_norm)


                
        
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

                if est_rho:
                    rr_coef = (rr_knot.squeeze(1) @ k_coef.transpose(-2, -1)).transpose(-2, -1)
                else :
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
                    
                    if est_rho:
                        pr_coef = (pr_knot.squeeze(1) @ k_coef.transpose(-2, -1)).transpose(-2, -1)
                    else: 
                        pr_coef = (pr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)
                    #pr_coef = (pr_knot @ k_coef.transpose(-2, -1)).transpose(-2, -1)
                    
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
                    if est_rho:
                        pr_coef = (pr_knot.squeeze(1) @ k_coef.transpose(-2, -1)).transpose(-2, -1)
                    else: 
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
  
        yhat = lev + (regressors * coef).sum(-1) #+ seas_term + meany

        obs_scale_base = pyro.sample("obs_scale_base", dist.Beta(2, 2)).unsqueeze(-1)
        # from 0.5 * sdy to sdy
        obs_scale = ((obs_scale_base * (1.0 - min_residuals_sd)) + min_residuals_sd) * sdy
        #print(yhat[..., which_valid])
        # with pyro.plate("response_plate", n_valid):
        #     pyro.sample("response",
        #                 dist.StudentT(dof, yhat[..., which_valid], obs_scale),
        #                 obs=response_tran[which_valid])


        #print(obs_scale)
        pyro.sample("response",
                    dist.StudentT(dof, loc = yhat[..., which_valid], scale= obs_scale).to_event(1),
                    obs=response_tran[which_valid])
        
        ## log likelihood
        d = (yhat[..., which_valid] - response_tran[which_valid])/obs_scale
        LL = - ((dof+1.0)/2.0)*torch.log(1+d/dof)
        dof_t = torch.zeros_like(LL) + dof 
        LL = LL + torch.lgamma((dof_t+1.0)/2.0) - 0.5*torch.log(dof_t*3.1415927410125732)- torch.lgamma(dof_t/2.0)
        LL = LL.sum(-1)

        

        lev_knot = lev_knot_tran + meany

        extra_out.update({
            'yhat': yhat + seas_term + meany,
            'lev': lev + meany,
            'lev_knot': lev_knot,
            'coef': coef,
            'coef_knot': coef_knot,
            'coef_init_knot': coef_init_knot,
            'obs_scale': obs_scale,
            'loglikelihood': LL,
            'rho_coef' : rho_coef,
        })
        return extra_out
