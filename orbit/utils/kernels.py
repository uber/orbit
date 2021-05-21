import numpy as np


def reduce_by_max(x, n=2):
    out = x.copy()
    out[np.argsort(x)[:-n]] = 0
    return out


# Gaussian-Kernel
# https://en.wikipedia.org/wiki/Kernel_smoother
def gauss_kernel_old(x, x_i, rho=1.0, alpha=1.0, n_reduce=-1, point_to_flatten=1):
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
    alpha_sq = alpha ** 2
    rho_sq_t2 = 2 * rho ** 2
    for n in range(N):
        if x[n] <= point_to_flatten:
            k[n, :] = alpha_sq * np.exp(-1 * (x[n] - x_i) ** 2 / rho_sq_t2)
        else:
            # last weights carried forward for future time points
            k[n, :] = alpha_sq * np.exp(-1 * (point_to_flatten - x_i) ** 2 / rho_sq_t2)

    if n_reduce > 0:
       k = np.apply_along_axis(reduce_by_max, axis=1, arr=k, n=n_reduce)

    k = k / np.sum(k, axis=1, keepdims=True)

    return k


def sandwich_kernel(x, x_i):
    """
    Parameters
    ----------
    x : array-like
        points required to compute kernel weight
    x_i : array-like
        reference points location used to compute correspondent distance of each entry points

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
    k = np.zeros((N, M), dtype=np.double)

    np_idx = np.where(x < x_i[0])
    k[np_idx, 0] = 1

    for m in range(M - 1):
        np_idx = np.where(np.logical_and(x >= x_i[m], x < x_i[m + 1]))
        total_dist = x_i[m + 1] - x_i[m]
        backward_dist = x[np_idx] - x_i[m]
        forward_dist = x_i[m + 1] - x[np_idx]
        k[np_idx, m] = forward_dist / total_dist
        k[np_idx, m + 1] = backward_dist / total_dist

    np_idx = np.where(x >= x_i[M - 1])
    k[np_idx, M - 1] = 1

    # TODO: it is probably not needed
    k = k / np.sum(k, axis=1, keepdims=True)

    return k


def parabolic_kernel(x, x_i):
    N = len(x)
    M = len(x_i)
    k = np.zeros((N, M), dtype=np.double)

    # boundary case
    np_idx = np.where(x < x_i[0])
    if len(np_idx) > 0:
        k[np_idx, 0] = 1

    for m in range(M - 1):
        np_idx = np.where(np.logical_and(x >= x_i[m], x < x_i[m + 1]))
        total_dist = x_i[m + 1] - x_i[m]
        backward_dist = x[np_idx] - x_i[m]
        forward_dist = x_i[m + 1] - x[np_idx]
        k[np_idx, m] = 0.75 * (1 - (backward_dist / total_dist) ** 2)
        k[np_idx, m + 1] = 0.75 * (1 - (forward_dist / total_dist) ** 2)

    # boundary case
    np_idx = np.where(x >= x_i[M - 1])
    if len(np_idx) > 0:
        k[np_idx, M - 1] = 1

    # TODO: it is probably not needed
    k = k / np.sum(k, axis=1, keepdims=True)

    return k


def is_neg(x):
    out = np.divide(x-np.absolute(x),-2*np.absolute(x),out=np.zeros_like(x), where=np.absolute(x)!=0.0 )
    out = np.absolute(out)
    return(out)

def gauss_kernel(x, x_i, rho=1.0, alpha=1.0, n_reduce=-1, point_to_flatten=1, norm = [1.0,1.0]):
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
    # this section is for making an asymetric kernal
    # sanitize the imputs 
    norm = np.divide(np.absolute(norm),max(np.absolute(norm)),out=np.ones_like(norm), where=max(np.absolute(norm))!=0.0 )
    norm = norm**2
    
    N = len(x)
    M = len(x_i)
    k = np.zeros((N, M), np.double)
    alpha_sq = alpha ** 2
    rho_sq_t2 = 2 * rho ** 2
    
    for n in range(N):
        if x[n] <= point_to_flatten:
           
            k[n, :] = alpha_sq * np.exp(-1 * (x[n] - x_i) ** 2 / 
                                       ( is_neg(x[n] - x_i)*rho_sq_t2*norm[0]  + (1-is_neg(x[n] - x_i))*rho_sq_t2**norm[1])) 
                
        else:
            # last weights carried forward for future time points
            k[n, :] = alpha_sq * np.exp(-1 * (point_to_flatten - x_i) ** 2 / 
                                       ( is_neg(x[n] - x_i)*rho_sq_t2*norm[0]  + (1-is_neg(x[n] - x_i))*rho_sq_t2**norm[1])) 

    if n_reduce > 0:
       k = np.apply_along_axis(reduce_by_max, axis=1, arr=k, n=n_reduce)

    k = k / np.sum(k, axis=1, keepdims=True)

    return k