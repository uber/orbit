import numpy as np

# Gaussian-Kernel
# https://en.wikipedia.org/wiki/Kernel_smoother
def gauss_kernel(x, x_i, rho=1.0, alpha=1.0):
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
    return W