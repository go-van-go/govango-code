import numpy as np
from recursivenodes.utils import npolys
from recursivenodes.polynomials import (
        jacobinorm2,
        jacobi,
        proriolkoornwinderdubinervandermonde as pkdv)
from recursivenodes.quadrature import (
        gaussjacobi,
        lobattogaussjacobi,
        simplexgausslegendre)


def test_gaussjacobi(k, tol=1.e-13):
    ''' Test Gauss-Jacobi quadrature for random weights a,b, verifying that
    the basis functions are orthonormal '''
    a = np.random.rand()
    b = np.random.rand()
    norms2 = np.array([jacobinorm2(i, a, b) for i in range(k+1)])

    x, w = gaussjacobi(k+1, a, b)
    V = np.ndarray((k+1, k+1))
    II = np.eye(k+1)
    for i in range(k+1):
        V[:, i] = jacobi(i, x, a, b)
    M = np.einsum('ki,k,kj->ij', V, w, V)/norms2.reshape((k+1, 1))
    err = np.max(np.abs(II - M).ravel())
    assert err < tol


def test_lobattogaussjacobi(k, tol=1.e-13):
    ''' Test Lobatto-Gauss-Jacobi quadrature for random weights a,b, verifying that
    the basis functions are orthonormal '''
    a = np.random.rand()
    b = np.random.rand()
    norms2 = np.array([jacobinorm2(i, a, b) for i in range(k+1)])

    x, w = lobattogaussjacobi(k+2, a, b)
    V = np.ndarray((k+2, k+1))
    II = np.eye(k+1)
    for i in range(k+1):
        V[:, i] = jacobi(i, x, a, b)
    M = np.einsum('ki,k,kj->ij', V, w, V)/norms2.reshape((k+1, 1))
    err = np.max(np.abs(II - M).ravel())
    assert err < tol


def test_simplexgausslegendre(d, k, tol=1.e-13):
    ''' Test simplex Gauss-Legendre quadrature, verifying that
    the Proriol-Koornwinder-Dubiner basis functions are orthonormal '''
    N = npolys(d, k)
    II = np.eye(N)
    x, w = simplexgausslegendre(d, k+1)
    V = pkdv(d, k, x.reshape(((k+1)**d, d)))
    M = np.einsum('ki,k,kj->ij', V, w.ravel(), V)
    err = np.max(np.abs(II - M).ravel())
    assert err < tol
