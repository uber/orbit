import numpy as np


def reduce_by_max(x, n=2):
    out = x.copy()
    out[np.argsort(x)[:-n]] = 0
    return out


# Gaussian-Kernel
# https://en.wikipedia.org/wiki/Kernel_smoother
def gauss_kernel(x, x_i, rho=1.0, alpha=1.0, n_reduce=-1):
    """
    x: points required to compute kernel weight
    x_i: reference points location used to compute correspondent distance of each entry points
    rho: smoothing parameter known as "length-scale" in gaussian process
    alpha: marginal standard deviation parameter in gaussian process; one should ignore in kernel regression (keep it = 1.0)
    b[deprecated]: radius or sometime named as (2*rho) that controls strength of covariance; the smaller the shorter raidus (dist. to negihbour)
    will take into effect
    return:
        a matrix with N x M such that
        N as the number of entry points
        M as the number of reference points
        matrix entries hold the value of weight of each element
    see also:
      1. https://mc-stan.org/docs/2_24/stan-users-guide/gaussian-process-regression.html
      2. https://en.wikipedia.org/wiki/Local_regression
    """
    N = len(x)
    M = len(x_i)
    W = np.zeros((N, M))
    alpha_sq = alpha ** 2
    rho_sq_t2 = 2 * rho ** 2
    for n in range(N):
        W[n, :] = alpha_sq * np.exp(-1 * (x[n] - x_i) ** 2 / rho_sq_t2)

    if n_reduce > 0:
       W = np.apply_along_axis(reduce_by_max, axis=1, arr=W, n=n_reduce)

    return W


def sandwich_kernel(x, x_i):
    """
    x: points required to compute kernel weight
    x_i: reference points location used to compute correspondent distance of each entry points
    rho: smoothing parameter known as "length-scale" in gaussian process
    alpha: marginal standard deviation parameter in gaussian process; one should ignore in kernel regression (keep it = 1.0)
    b[deprecated]: radius or sometime named as (2*rho) that controls strength of covariance; the smaller the shorter raidus (dist. to negihbour)
    will take into effect
    return:
        a matrix with N x M such that
        N as the number of entry points
        M as the number of reference points
        matrix entries hold the value of weight of each element
    see also:
      1. https://mc-stan.org/docs/2_24/stan-users-guide/gaussian-process-regression.html
      2. https://en.wikipedia.org/wiki/Local_regression
    """
    N = len(x)
    M = len(x_i)
    W = np.zeros((N, M))

    np_idx = np.where(x < x_i[0])
    W[np_idx, 0] = 1

    for m in range(M - 1):
        np_idx = np.where(np.logical_and(x >= x_i[m], x < x_i[m + 1]))
        total_dist = x_i[m + 1] - x_i[m]
        backward_dist = x[np_idx] - x_i[m]
        forward_dist = x_i[m + 1] - x[np_idx]
        W[np_idx, m] = forward_dist / total_dist
        W[np_idx, m + 1] = backward_dist / total_dist

    np_idx = np.where(x >= x_i[M - 1])
    W[np_idx, M - 1] = 1

    return W