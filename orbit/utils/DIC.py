import numpy as np
import math
import copy 
from scipy import stats
from math import pi
import scipy.special as sc
import time


def ll_hood(d, dof, loc, scale):
    ## log likelihood
    d = (d - loc)/scale
    LL = - ((dof+1.0)/2.0)*np.log(1+d/dof)
    LL = LL + sc.loggamma((dof+1.0)/2.0) - 0.5*np.log(dof*3.1415927410125732)- sc.loggamma(dof/2.0)
    LL = LL.sum(-1)
    return(LL)

def gauss_kernel(x, x_i, rho=1.0, alpha=1.0, n_reduce=-1, point_to_flatten=1,  norm = [1.0,1.0]):
    """
    Parameters
    ----------
    x : array-like
        points required to compute kernel weight
    x_i : array-like
        reference points location used to compute correspondent distance of each entry points
    rho : float
        smoothing parameter known as "length-scale" in gaussian process
    alpha : float
        marginal standard deviation parameter in gaussian process; one should use 1 in kernel regression
    point_to_flatten : float
        the time point starting to flatten the weights; default is 1 for normalized time points

    Returns
    -------
    np.ndarray
        2D array with size N x M such that
        N as the number of entry points
        M as the number of reference points
        matrix entries hold the value of weight of each element

    See Also
    --------
      1. https://mc-stan.org/docs/2_24/stan-users-guide/gaussian-process-regression.html
      2. https://en.wikipedia.org/wiki/Local_regression
    """
    N = len(x)
    M = len(x_i)
    k = np.zeros((N, M), np.double)
    #junk_for_print = np.zeros((N, M), np.double)
    alpha_sq = alpha ** 2
    rho_sq_t2 = 2 * (rho ** 2)
    for n in range(N):
        #junk_for_print[n, :] = (x[n] - x_i)**2
        if x[n] <= point_to_flatten:
            k[n, :] = alpha_sq * np.exp(-1 * (x[n] - x_i) ** 2 / rho_sq_t2)
        else:
            # last weights carried forward for future time points
            k[n, :] = alpha_sq * np.exp(-1 * (point_to_flatten - x_i) ** 2 / rho_sq_t2)

    if n_reduce > 0:
       k = np.apply_along_axis(reduce_by_max, axis=1, arr=k, n=n_reduce)
    
    k = k / np.sum(k, axis=1, keepdims=True)

    return k


def deviance_cent(mod, data, dof):
    # get the center model
    pars = mod._posterior_samples.keys()
    samps = mod._posterior_samples
    cents = copy.deepcopy(samps)
    
    for p in pars:
        #print(p)
        # hack but works 
        if len(samps[p].shape) == 1:
            cents[p][0] = np.mean(samps[p],axis=0)
        if len(samps[p].shape) == 2:
            cents[p][0,:] = np.mean(samps[p],axis=0)    
        if len(samps[p].shape) == 3:
            cents[p][0,:,:] = np.mean(samps[p],axis=0)               
    if mod.est_rho:
        n_obs = data.shape[0]
        mod.k_coef = gauss_kernel(x = np.arange(1, n_obs + 1) / n_obs, 
                              x_i = mod._knots_tp_coefficients, 
                              rho=cents['rho_coef'][0], 
                              )
    # make the preds     
    yhat = mod._predict(df=data,posterior_estimates=cents)
    yhat = yhat['prediction'][0,:]
    
    # likilihood 
    LL = ll_hood(d = mod.response, dof= dof, loc = yhat, scale = cents['obs_scale'][0])
    # make the device 
    dev = -2*LL
    return(dev)
    
def DIC(mod, data, dof):
    D_MAP = deviance_cent(mod = mod, data = data, dof = dof)
    D_AVG = -2.0*np.mean(mod._posterior_samples['loglikelihood'])
    Pd = D_AVG - D_MAP
    DIC_val = D_AVG + Pd
    return(DIC_val)


