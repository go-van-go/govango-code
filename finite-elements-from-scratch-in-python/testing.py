import numpy as np
import pdb
from scipy.special import gamma
from scipy.linalg import eig
from scipy.special import jacobi
from scipy.special import eval_jacobi


def jacobi_p(x, alpha, beta, N):
    # Turn points into row if needed.
    xp = x

    # PL will carry our values for the jacobi polynomial
    PL = np.zeros((N+1, len(x)))

    # initialize values P_0(x)
    gamma0 = 2 ** (alpha + beta + 1) / (alpha + beta + 1) * gamma(alpha + 1) * \
             gamma(beta + 1) / gamma(alpha + beta + 1)
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    # return if N = 0
    if N == 0:
        P = PL[0, :]
        return P

    # initialize value P_1(x)
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)

    # return if N = 1
    if N == 1:
        P = PL[N, :]
        return P

    # Repeat value in recurrence.
    a_old = 2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))

    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        a_new = 2 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) *
                                       (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        PL[i + 1, :] = 1 / a_new * (-a_old * PL[i-1, :] + (xp - b_new) * PL[i, :])
        a_old = a_new

    #P = np.reshape(PL[N, :], (np.shape(PL[N, :])[0], 1)).T
    P = PL[N, :]
    return P

import numpy as np
from scipy.special import eval_jacobi, gamma

def normalized_jacobi(x, n, alpha, beta):
    """
    Compute the normalized Jacobi polynomial of degree n at points x.
    """
    # Compute the unnormalized Jacobi polynomial using scipy
    P_n = eval_jacobi(n, alpha, beta, x)
    
    # Compute the normalization constant gamma_n
    numerator = 2 ** (alpha + beta + 1) * gamma(n + alpha + 1) * gamma(n + beta + 1)
    denominator = (2 * n + alpha + beta + 1) * gamma(n + alpha + beta + 1) * gamma(n + 1)
    gamma_n = numerator / denominator
    
    # Normalize the polynomial
    P_n_normalized = P_n / np.sqrt(gamma_n)
    
    return P_n_normalized


def test_jacobi_functions(num_tests=100, tol=1e-10):
    """
    Test if jacobi_p and normalized_jacobi return the same values for random inputs.
    
    Parameters:
        num_tests (int): Number of random tests to perform.
        tol (float): Tolerance for comparing the results.
    """
    np.random.seed(36)  # For reproducibility
    for _ in range(num_tests):
        # Generate random inputs
        x = np.random.uniform(-1, 1, size=np.random.randint(1, 100))  # Random x array
        #alpha = np.random.uniform(-0.99, 60)  # Random alpha > -1
        alpha=1
        #beta = np.random.uniform(-0.99, 60)   # Random beta > -1
        beta=1
        n = np.random.randint(0, 30)         # Random degree n >= 0
        
        # Compute values from both functions
        P_custom = jacobi_p(x, alpha, beta, n)
        P_scipy = normalized_jacobi(x, n, alpha, beta)
        
        # Check if the results are close
        try:
            np.testing.assert_allclose(P_custom, P_scipy, rtol=tol, atol=tol)
        except AssertionError as e:
            print(f"Test failed for: alpha={alpha}, beta={beta}, n={n}, x={x}")
            print(f"Custom function result: {P_custom}")
            print(f"Scipy function result: {P_scipy}")
            raise e
    
    print(f"All {num_tests} tests passed!")

# Run the test
test_jacobi_functions(num_tests=10000)
