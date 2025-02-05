import pytest
import numpy as np
from recursivenodes.polynomials import (
        _eval_jacobi,
        jacobi,
        jacobider,
        proriolkoornwinderdubiner as pkd,
        proriolkoornwinderdubinergrad as pkdgrad,
        proriolkoornwinderdubinerhess as pkdhess,
        proriolkoornwinderdubinervandermonde as pkdv,
        proriolkoornwinderdubinervandermondegrad as pkdvgrad,
        proriolkoornwinderdubinervandermondehess as pkdvhess)
from recursivenodes.utils import npolys
from utils import fdtest


def test__eval_jacobi(k):
    '''compare _eval_jacobi with scipy.special.eval_jacobi'''
    scipyspecial = pytest.importorskip("scipy.special")
    eval_jacobi = scipyspecial.eval_jacobi
    x = np.random.rand(3, 2)
    a = np.random.rand()
    b = np.random.rand()
    _p = _eval_jacobi(k, a, b, x, None)
    p = eval_jacobi(k, a, b, x, None)
    err = np.abs(_p - p)
    assert err.max() < 100. * np.finfo(err.dtype).eps


def test_jacobi_der(k, eps=1.e-5, step=0.5, tol=1.e-2, rtol=1.e-16):
    '''compare the analytically computed expression for the derivative of
    a Jacobi polynomial to a finite difference approximation computed at a
    random point'''
    x = np.random.rand() * 2 - 1
    dx = np.random.rand()
    a = np.random.rand()
    b = np.random.rand()
    rate = fdtest(lambda _x: jacobi(k, _x, a, b),
                  lambda _x: jacobider(k, _x, a, b, k=1),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)


def test_jacobi_hess(k, eps=1.e-5, step=0.5, tol=1.e-2, rtol=1.e-16):
    '''compare the analytically computed expression for the second derivative
    of a Jacobi polynomial to a finite difference approximation computed at a
    random point'''
    x = np.random.rand() * 2 - 1
    dx = np.random.rand()
    a = np.random.rand()
    b = np.random.rand()
    rate = fdtest(lambda _x: jacobider(k, _x, a, b, k=1),
                  lambda _x: jacobider(k, _x, a, b, k=2),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)


def test_pkd_grad(d, k, index=None, eps=1.e-5, step=0.5, tol=1.e-2,
                  rtol=1.e-16):
    '''compare the analytically computed expression for the gradient of a
    Proriol-Koornwinder-Dubiner basis function with random index to a finite
    difference approximation computed at random points'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    x[0, :] = -1
    x[0, -1] = 1
    dx = np.random.rand(5, d)
    y = np.random.rand(5)
    if index is None:
        index = [0] * d
        for j in range(d-1):
            index[j] = np.random.randint(k-sum(index[0:j])+1)
        index[-1] = k-sum(index[0:-1])
        index = tuple(index)
    rate = fdtest(lambda _x: y.dot(pkd(d, index, _x)),
                  lambda _x: (y.reshape(5, 1) * pkdgrad(d, index, _x)),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)


def test_pkd_hess(d, k, index=None, eps=1.e-5, step=0.5, tol=1.e-2,
                  rtol=1.e-16):
    '''compare the analytically computed expression for the Hessian of a
    Proriol-Koornwinder-Dubiner basis function with random index to a finite
    difference approximation computed at random points'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    x[0, :] = -1
    x[0, -1] = 1
    dx = np.random.rand(5, d)
    z = np.random.rand(5, d)
    if index is None:
        index = [0] * d
        for j in range(d-1):
            index[j] = np.random.randint(k-sum(index[0:j])+1)
        index[-1] = k-sum(index[0:-1])
        index = tuple(index)
    rate = fdtest(lambda _x: z.ravel().dot(pkdgrad(d, index, _x).ravel()),
                  lambda _x: np.einsum('ij,ijk->ik', z, pkdhess(d, index, _x)),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)


def test_pkdv_grad(d, k, C=False, eps=1.e-5, step=0.5, tol=1.e-2,
                   rtol=1.e-16):
    '''compare the analytically computed expression for the gradient of a
    Proriol-Koornwinder-Dubiner Vandermonde matrix (potentially pre-multiplied
    by a matrix C) to a finite difference approximation computed at random
    points'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    x[0, :] = -1
    x[0, -1] = 1
    dx = np.random.rand(5, d)
    N = npolys(d, k)
    if C:
        C = np.random.rand(N, 3)
        y = np.random.rand(5, 3)
    else:
        C = None
        y = np.random.rand(5, N)
    rate = fdtest(lambda _x: y.ravel().dot(pkdv(d, k, _x, C=C).ravel()),
                  lambda _x: np.einsum('ij,ijk->ik',
                                       y, pkdvgrad(d, k, _x, C=C)),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)


def test_pkdv_hess(d, k, C, eps=1.e-5, step=0.5, tol=1.e-2,
                   rtol=1.e-16):
    '''compare the analytically computed expression for the Hessian of a
    Proriol-Koornwinder-Dubiner Vandermonde matrix (potentially pre-multiplied
    by a matrix C) to a finite difference approximation computed at random
    points'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    x[0, :] = -1
    x[0, -1] = 1
    dx = np.random.rand(5, d)
    N = npolys(d, k)
    if C:
        C = np.random.rand(N, 3)
        y = np.random.rand(5, 3, d)
    else:
        C = None
        y = np.random.rand(5, N, d)
    rate = fdtest(lambda _x: np.einsum('ijk,ijk', y, pkdvgrad(d, k, _x, C=C)),
                  lambda _x: np.einsum('ijk,ijkl->il',
                                       y, pkdvhess(d, k, _x, C=C)),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)
