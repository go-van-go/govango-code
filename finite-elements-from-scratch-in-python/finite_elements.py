import numpy as np
from pathlib import Path


class LagrangeElement:
    """Lagrange finite element defined on a triangle and
      tetrahedron"""
    # get the base directory where the script is located, including the 'tabulated_nodes' folder
    base_dir = Path(__file__).parent / "tabulated_nodes"

    # load the nodes once as a class attribute
    _line_nodes = np.load(base_dir / "line_nodes.npz")
    _triangle_nodes = np.load(base_dir / "triangle_nodes.npz")
    _tetrahedron_nodes = np.load(base_dir / "tetrahedron_nodes.npz")

    def __init__(self, d, n):
        self.d = d  # dimension
        self.n = n  # polynomial order
        self.nodes_per_element = (n + 1) * (n + 2) * (n + 3) // 6
        self.nodes_per_face = (n + 1) * (n + 2) // 2
        self.num_faces = 0  # no. faces per element
        self.nodes = np.array([])
        self.r = np.array([])
        self.s = np.array([])
        self.t = np.array([])
        self.vertices = np.array([])
        self.NODE_TOLERANCE = 1e-7  # tolerance to find face nodes

        self._get_nodes_and_vertices()

    def _get_nodes_and_vertices(self):
        """ get tabulated data for nodes, and mesh data for vertices"""
        n = self.n
        if n < 30:
            if self.d == 1:
                # line element
                self.num_faces = 2
                self.nodes = self._line_nodes[str(n)]
                self.r =  self.nodes # x coordinate of each node 
                self.vertices = np.array([0,1])
            elif self.d == 2:
                # triangle
                self.num_faces = 3
                self.nodes = self._triangle_nodes[str(n)]
                self.r =  self.nodes[:,0] # x coordinate of each node 
                self.s =  self.nodes[:,1]# y coordinate of each node
                self.vertices = np.array([[0, 0],
                                          [1, 0],
                                          [0, 1]])
            elif self.d == 3:
                # tetrahedron
                self.num_faces = 4
                self.nodes = self._tetrahedron_nodes[str(n)]
                self.r =  self.nodes[:,0] # x coordinate of each node 
                self.s =  self.nodes[:,1]# y coordinate of each node
                self.t =  self.nodes[:,2] # z coordinate of each node
                self.vertices = np.array([[0, 0, 0],
                                          [1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]])
        else:
            raise Exception(f"No precomputed nodes found for d={d}, n={n}")

    def eval_basis_function_3d(self, r, s, t, i, j, k):
        """ evaluate 3D orthonormal basis functions"""
        # transfer to the simplified coordinates for the Jacobi polynomials
        a, b, c = _rst_to_abc(r,s,t)
        # return the evaluated basis functions
        return self.orthonormal_polynomial_3d(a, b, c, i, j, k)

    def eval_basis_function_2d(self, r, s, i, j):
        """ evaluate 2D orthonormal basis functions"""
        # transfer to the simplified coordinates for the Jacobi polynomials
        a, b = _rs_to_ab(r,s)
        # return the evaluated basis functions
        return self.orthonormal_polynomial_2d(a, b, i, j)

    def eval_basis_function_3d_gradient(self, r, s, t, i, j, k):
        """ evaluate gradient of 3D basis functions """
        # transfer to the simplified coordinates for the Jacobi polynomials
        a, b, c = _rst_to_abc(r,s,t)
        # return the evaluated basis functions
        return self.orthonormal_polynomial_3d_derivative(a, b, c, i, j, k)

       
    def _rst_to_abc(r, s, t):
        """ transfer from (r,s,t) coordinates to (a,b,c) which are used to evaluate the
        jacobi polynomials in our orthonormal basis """
        Np = len(r)
        a = np.zeros(Np)
        b = np.zeros(Np)
        c = np.zeros(Np)
        
        for n in range(Np):
            if s[n] + t[n] != 0:
                a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1
            else:
                a[n] = -1

            if t[n] != 1:
                b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1
            else:
                b[n] = -1

            c[n] = t[n]
        
        return a, b, c

    def _rs_to_ab(r, s):
        """ map from (r,s) coordinates on an element face to (a,b) which are used to evaluate the
            jacobi polynomials in our orthonormal basis """
        Np = len(r)
        a = np.zeros(Np)
        for n in range(Np):
            if s[n] != 1:
                a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1
            else:
                a[n] = -1
        b = s
        return a, b


    def orthonormal_polynomial_3d(a, b, c, i, j, k):
        """ evaulated orthonormal basis function polynomials """
        h1 = self.normalized_jacobi(a, 0, 0, i)
        h2 = self.normalized_jacobi(b, 2*i+1, 0, j)
        h3 = self.normalized_jacobi(c, 2*(i+j)+2, 0, k)
        
        P = 2 * np.sqrt(2) * h1 * h2 * ((1 - b) ** i) * h3 * ((1 - c) ** (i + j))
        return P


    def orthonormal_polynomial_2d(a, b, i, j):
        """ Evaluate 2D orthonormal polynomial on simplex at (a,b) of order (i,j) """
        h1 = self.normalized_jacobi(a, 0, 0, i)
        h2 = self.normalized_jacobi(b, 2 * i + 1, 0, j)

        P = np.sqrt(2.0) * h1 * h2 * (1 - b) ** i
        return P


    def orthonormal_polynomial_3d_derivative(a, b, c, i, j, k):
        """ Return the derivatives of the modal basis (id,jd,kd) on the 3D simplex at (a,b,c)"""
        
        fa = normalized_jacobi(a, 0, 0, i)
        dfa = normalized_jacobi_gradient(a, 0, 0, i)
        gb = normalized_jacobi(b, 2*i+1, 0, j)
        dgb = normalized_jacobi_gradient(b, 2*i+1, 0, j)
        hc = normalized_jacobi(c, 2*(i+j)+2, 0, k)
        dhc = normalized_jacobi_gradient(c, 2*(i+j)+2, 0, k)
        
        # calculate r derivative Vr
        Vr = dfa * (gb * hc)
        if i > 0:
            Vr *= (0.5 * (1 - b)) ** (i - 1)
        if i + jd > 0:
            Vr *= (0.5 * (1 - c)) ** (i + j - 1)
        
        # calculate s derivative Vs
        Vs = 0.5 * (1 + a) * Vr
        tmp = dgb * ((0.5 * (1 - b)) ** i)
        if i > 0:
            tmp += (-0.5 * i) * (gb * ((0.5 * (1 - b)) ** (i - 1)))
        if i + j > 0:
            tmp *= (0.5 * (1 - c)) ** (i + j - 1)
        tmp = fa * (tmp * hc)
        Vs += tmp
        
        # calculate t derivative Vt
        Vt = 0.5 * (1 + a) * Vr + 0.5 * (1 + b) * tmp
        tmp = dhc * ((0.5 * (1 - c)) ** (i + j))
        if i + j > 0:
            tmp -= 0.5 * (i + j) * (hc * ((0.5 * (1 - c)) ** (i + j - 1)))
        tmp = fa * (gb * tmp)
        tmp *= (0.5 * (1 - b)) ** i
        Vt += tmp
        
        # normalize
        Vr *= 2 ** (2*i+j+1.5)
        Vs *= 2 ** (2*i+j+1.5)
        Vt *= 2 ** (2*i+j+1.5)
        
        return Vr, Vs, Vt


    def normalized_jacobi(x, n, alpha, beta):
        """
        Compute the normalized Jacobi polynomial of degree n at points x.
        """
        # compute the unnormalized Jacobi polynomial using scipy
        P_n = eval_jacobi(n, alpha, beta, x)
        
        # compute the normalization constant gamma_n
        numerator = 2 ** (alpha + beta + 1) * gamma(n + alpha + 1) * gamma(n + beta + 1)
        denominator = (2 * n + alpha + beta + 1) * gamma(n + alpha + beta + 1) * gamma(n + 1)
        gamma_n = numerator / denominator
        
        # normalize the polynomial
        P_n_normalized = P_n / np.sqrt(gamma_n)
        
        return P_n_normalized

 
    def normalized_jacobi_gradient(r, alpha, beta, N):
        dP = np.zeros(len(r))
        if N == 0:
            dP[:] = 0.0
        else:
            dP = np.sqrt(N * (N + alpha + beta + 1)) * \
                normalized_jacobi(r, alpha + 1, beta + 1, N - 1)
        

if __name__ == "__main__":
    pass
