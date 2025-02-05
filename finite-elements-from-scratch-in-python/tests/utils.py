
import numpy as np


def fdtest(f, df, x, dx, eps, step, rtol):
    '''Finite difference test of derivative functions.

    Args:
        f (callable): Function taking one 1D ndarray that returns a float.
        df (callable): Function taking one 1D ndarray that returns an ndarray
            with the same shape.
        x (ndarray): Initial point where `f` and `df` are called.
        dx (ndarray): The variation in `x`.
        eps (float): The scale of the perturbation in the variation direction.
        step (float): The multiplier used to produce a second perturbation in
            the `dx` direction.
        rtol (float): The tolerance at which a derivative is declared to be
            exact.

    Returns:
        float: a numerical estimate of the rate of convergence of the finite
        difference estimate to the derivative from `df`.
    '''
    dx = np.array(dx)
    fval = f(x)
    fder = np.array(df(x))
    diff = []
    eps = [eps, eps * step]
    for e in eps:
        fpred = fval + np.array(fder.ravel().dot(dx.ravel()) * e)
        xvar = x + dx * e
        fvar = f(xvar)
        diff.append(fpred - fvar)
    if max(np.abs(diff[0]), np.abs(diff[1])) < rtol:
        rate = np.inf
    else:
        rate = np.log(np.abs(diff[0]/diff[1])) / np.log(eps[0]/eps[1])
    return rate
