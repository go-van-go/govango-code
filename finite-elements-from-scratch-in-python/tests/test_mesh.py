import pytest
import numpy as np
from wave_simulator.mesh import Mesh3d
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators

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
