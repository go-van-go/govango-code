import pytest
import numpy as np

pytest.importorskip("scipy")
from recursivenodes.lebesgue import (
        lebesgue,
        lebesguegrad,
        lebesguemin,
        lebesgueminhess,
        lebesguemax,
        lebesguemax_reference)
from recursivenodes.nodes import (
        equispaced,
        blyth_luo_pozrikidis,
        warburton,
        recursive)
from utils import fdtest


def test_lebesguegrad(d, k, eps=1.e-5, step=0.5, tol=1.e-2, rtol=1.e-16):
    '''Test that the analytically computed gradient of the Lebesgue function
    at random points in the biunit simplex matches a finite difference
    approximation'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    dx = np.random.rand(5, d)
    nodes = equispaced(d, k, domain='biunit')
    y = np.random.rand(5,)
    rate = fdtest(lambda _x: y.dot(lebesgue(_x, d, k, nodes)),
                  lambda _x: y.reshape(5, 1) * lebesguegrad(_x, d, k, nodes),
                  x, dx, eps, step, rtol)
    assert rate > (2.-tol)


def test_lebesguemingrad(d, k, eps=1.e-5, step=0.5, tol=1.e-2, rtol=1.e-16):
    '''Test that the analytically computed gradient of the negative sum of
    Lebesgue functions at multiple random points in the biunit simplex matches
    a finite difference approximation'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    dx = np.random.rand(5, d)
    nodes = equispaced(d, k, domain='biunit')
    rate = fdtest(lambda _x: lebesguemin(_x, d, k, nodes)[0],
                  lambda _x: lebesguemin(_x, d, k, nodes)[1],
                  x.reshape((5*d,)), dx.reshape((5*d,)), eps, step, rtol)
    assert rate > (2.-tol)


def test_lebesgueminhess(d, k, eps=1.e-4, step=0.5, tol=1.e-2, rtol=1.e-16):
    '''Test that the analytically computed Hessian of the negative sum of
    Lebesgue functions at multiple random points in the biunit simplex matches
    a finite difference approximation'''
    x = np.random.rand(5, d) * 2 - 1
    x[:, 0:(d-1)] = (x[:, 0:(d-1)] + 1)*(1 - x[:, d-1]).reshape((5, 1))/2 - 1
    dx = np.random.rand(5, d)
    nodes = equispaced(d, k, domain='biunit')
    y = np.random.rand(5 * d)
    rate = fdtest(lambda _x: y.dot(lebesguemin(_x, d, k, nodes)[1]),
                  lambda _x: lebesgueminhess(_x, d, k, nodes).dot(y),
                  x.reshape((5*d,)), dx.reshape((5*d,)), eps, step, rtol)
    assert rate > (2.-tol)


def test_lebesguemax(d_ref, k_ref, nodes, tol=1.e-2):
    '''Test the computed values of the Lebesgue constant for different nodes
    against precomputed values for regression'''
    if nodes == 'equispaced':
        nodes_x = equispaced(d_ref, k_ref, domain='biunit')
    elif nodes == 'blyth_luo_pozrikidis':
        nodes_x = blyth_luo_pozrikidis(d_ref, k_ref, domain='biunit')
    elif nodes == 'warburton':
        nodes_x = warburton(d_ref, k_ref, domain='biunit')
    elif nodes == 'recursive':
        nodes_x = recursive(d_ref, k_ref, domain='biunit')
    else:
        raise RuntimeError
    l_comp = lebesguemax(d_ref, k_ref, nodes_x)[0]
    l_ref = lebesguemax_reference(d_ref, k_ref, nodes)
    assert np.abs(l_comp - l_ref) < l_ref * tol
