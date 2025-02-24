import pytest
import numpy as np
from mesh import Mesh3d
from finite_elements import LagrangeElement
from reference_element_operators import ReferenceElementOperators
from scipy.special import roots_jacobi

@pytest.fixture(scope="module")
def finite_element():
    return LagrangeElement(d=3, n=3)

@pytest.fixture(scope="module")
def operators(finite_element):
    return ReferenceElementOperators(finite_element)

def test_vandermonde_3d_well_conditioned(operators):
    """Check that the Vandermonde 3D matrix is well-conditioned."""
    V = operators.vandermonde_3d
    cond_number = np.linalg.cond(V)
    
    assert cond_number < 1e6, f"Vandermonde matrix is poorly conditioned: {cond_number}"

def test_normalized_jacobi(finite_element):
    """Test that the normalized Jacobi polynomials have unit L2 norm."""
    x, w = roots_jacobi(10, 0, 0)  # 10-point Gauss quadrature
    for n in range(9):  # Test for first few degrees
        P_n = finite_element.normalized_jacobi(x, n, 0, 0)
        integral = np.sum(P_n**2 * w)  # Approximate L2 norm
        assert np.isclose(integral, 1.0, atol=1e-5), f"Normalized Jacobi polynomial of degree {n} is not unit norm"
