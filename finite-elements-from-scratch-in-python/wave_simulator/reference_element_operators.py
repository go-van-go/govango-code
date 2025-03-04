import numpy as np
from math import comb
from wave_simulator.visualizing import *
from wave_simulator.finite_elements import LagrangeElement

class ReferenceElementOperators:

    def __init__(self, finite_element: LagrangeElement):
        self.reference_element =  finite_element
        num_faces = finite_element.num_faces
        nodes_per_cell = self.reference_element.nodes_per_cell
        nodes_per_face = self.reference_element.nodes_per_face
        self.vandermonde_2d = np.zeros((nodes_per_face, nodes_per_face))
        self.vandermonde_3d = np.zeros((nodes_per_cell, nodes_per_cell))
        self.vandermonde_3d_r_derivative = np.zeros((nodes_per_cell, nodes_per_cell))
        self.vandermonde_3d_s_derivative = np.zeros((nodes_per_cell, nodes_per_cell))
        self.vandermonde_3d_t_derivative = np.zeros((nodes_per_cell, nodes_per_cell))
        self.inverse_vandermonde_3d = np.zeros((nodes_per_cell,nodes_per_cell))
        self.mass_matrix = np.zeros((nodes_per_cell,nodes_per_cell))
        self.r_differentiation_matrix = np.zeros((nodes_per_cell, nodes_per_cell))
        self.s_differentiation_matrix = np.zeros((nodes_per_cell, nodes_per_cell))
        self.t_differentiation_matrix = np.zeros((nodes_per_cell, nodes_per_cell))
        self.lift_matrix = np.zeros((nodes_per_cell, num_faces * nodes_per_face))
        self._calculate_element_operators()
        

    def _calculate_element_operators(self):
        r = self.reference_element.r
        s = self.reference_element.s
        t = self.reference_element.t
        #self._build_vandermonde_3d()
        self.vandermonde_3d = self._build_vandermonde(self.reference_element.d, self.reference_element.nodes)
        self._build_inverse_vandermonde_3d()
        self._build_vandermonde_3d_gradient()
        self._build_mass_matrix()
        self._build_differentiation_matrices()
        self._build_lift_matrix()
        
        
    def old_build_vandermonde_3d(self, r, s, t):
        """ create 3D vandermonde matrix"""

        # initialize the 3D Vandermonde Matrix
        vandermonde_matrix = self.vandermonde_3d
        # get orthonormal basis
        eval_basis_function_3d = self.reference_element.eval_basis_function_3d
        # get polynomial order of finite element
        n = self.reference_element.n 
        
        # build the Vandermonde matrix
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    vandermonde_matrix[:, column_index] = eval_basis_function_3d(r, s, t, i, j, k)
                    column_index += 1

        # store result
        self.vandermonde_3d = vandermonde_matrix
    
    def _multiindex_equal(self, d, k):
        """A generator for :math:`d`-tuple multi-indices whose sum is :math:`k`.

        Args:
        d (int): The length of the tuples
        k (int): The sum of the entries in the tuples

        Yields:
        tuple: tuples of length `d` whose entries sum to `k`, in lexicographic
        order.

        Example:
        >>> for i in multiindex_equal(3, 2): print(i)
        (0, 0, 2)
        (0, 1, 1)
        (0, 2, 0)
        (1, 0, 1)
        (1, 1, 0)
        (2, 0, 0)
        """
        if d <= 0:
            return
        if k < 0:
            return
        for i in range(k):
            for a in self._multiindex_equal(d-1, k-i):
                yield (i,) + a
        yield (k,) + (0,)*(d-1)


    def _multiindex_up_to(self, d, k):
        """A generator for :math:`d`-tuple multi-indices whose sum is at most
        :math:`k`.

        Args:
        d (int): The length of the tuples
        k (int): The maximum sum of the entries in the tuples

        Yields:
        tuple: tuples of length `d` whose entries sum to `k`, in lexicographic
        order.

        Example:
        >>> for i in multiindex_up_to(3, 2): print(i)
        (0, 0, 0)
        (0, 0, 1)
        (0, 0, 2)
        (0, 1, 0)
        (0, 1, 1)
        (0, 2, 0)
        (1, 0, 0)
        (1, 0, 1)
        (1, 1, 0)
        (2, 0, 0)
        """
        for a in self._multiindex_equal(d+1, k):
            yield a[0:d]

    def _npolys(self, d, k):
        """The number of polynomials up to a degree :math:`k` in :math:`d` dimensions.

        Args:
        d (int): The dimension of the space.
        k (int): The polynomial degree.

        Returns:
        int: :math:`\\binom{k+d}{d}`.

        Example:
        >>> npolys(3, 2)
        10
        """
        return comb(k+d, d)



    def _build_vandermonde(self, d, x):
        """ create 3D vandermonde matrix"""
    #def proriolkoornwinderdubinervandermonde(d, n, x, out=None, C=None):
        '''Evaluation of the Vandermonde matrix of the Proriol-Koornwinder-Dubiner
        (PKD) polynomials.

        Args:
        d (int): The spatial dimension.
        n (int): The maximum degree of polynomials to include in the
            Vandermonde matrix.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial Vandermonde matrix.
        out (ndarray, optional): Shape (`N`, `\\binom{n+d}{d}`) array to hold
            output (or, if ``C`` is given, shape (`N`, `M`), where `M` is the
            number of columns of `C`).
        C (ndarray, optional): Shape (`\\binom{n+d}{d}`, `M`) matrix to
            multiply the PDK Vandermonde matrix by on the right.

        Returns:
        ndarray: Shape (`N`, `\\binom{n+d}{d}`), evaluation of the PKD
        polynomials with leading monomial degree at most `n` at the points `x`.
        The columns will index the PKD polynomials in lexicographic degree of
        their leading monomial exponents.

        Or, if `C` is given, shape (`N`, `M`), the product of the Vandermonde
        matrix with `C`.
        '''
        out = None
        C = None
        n = self.reference_element.n
        N = self._npolys(d, n)
        if out:
            V = out
        else:
            if C is None:
                V = np.ndarray(x.shape[0:-1]+(N,))
            else:
                V = np.ndarray(x.shape[0:-1]+(C.shape[1],))
        if not (C is None):
            V[:] = 0.
        for (k, i) in enumerate(self._multiindex_up_to(d, n)):
            if C is None:
                V[:, k] = self.reference_element.proriolkoornwinderdubiner(d, i, x)
            else:
                V[:, :] += np.einsum('i,j->ij',
                                     self.reference_element.proriolkoornwinderdubiner(d, i, x),
                                     C[k, :])
        return V



    def old_build_vandermonde_2d(self, r, s):
        """ create 2D vandermonde matrix to evaluate flux at faces of each element"""
        
        # initiate vandermonde matrix
        vandermonde_matrix = self.vandermonde_2d
        
        # get basis function
        eval_basis_function_2d = self.reference_element.eval_basis_function_2d

        # get polynomial order of finite element
        n = self.reference_element.n 

        # build the Vandermonde matrix
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                vandermonde_matrix[:, column_index] = eval_basis_function_2d(r, s, i, j)
                column_index += 1

        # return result
        return vandermonde_matrix
                

    def _build_inverse_vandermonde_3d(self):
        """ invert the 3D Vandermonde Matrix """
        self.inverse_vandermonde_3d = np.linalg.inv(self.vandermonde_3d)


    def old_build_vandermonde_3d_gradient(self, r, s, t):
        """ Build gradient (Vr, Vs, Vt) of Vandermonde matrix"""
        # initialize Vandermonde derivative matrices
        Vr = self.vandermonde_3d_r_derivative
        Vs = self.vandermonde_3d_s_derivative
        Vt = self.vandermonde_3d_t_derivative
        
        # get basis function
        eval_basis_function_3d_gradient = self.reference_element.eval_basis_function_3d_gradient

        # get polynomial order of finite element
        n = self.reference_element.n 
        
        # build Vandermonde derivative matrices
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    Vr[:, column_index], Vs[:, column_index], Vt[:, column_index] = \
                        eval_basis_function_3d_gradient(r, s, t, i, j, k)
                    column_index += 1
                    
        # store result
        self.vandermonde_3d_r_derivative = Vr
        self.vandermonde_3d_s_derivative = Vs
        self.vandermonde_3d_t_derivative = Vt
 
    def _build_vandermonde_3d_gradient(self):
        # def proriolkoornwinderdubinervandermondegrad(d, n, x, out=None, C=None,
        #                                         work=None, both=False):
        '''Evaluation of the gradient of the Vandermonde matrix of the
        Proriol-Koornwinder-Dubiner (PKD) polynomials.

        Args:
            d (int): The spatial dimension.
            n (int): The maximum degree of polynomials to include in the
                Vandermonde matrix.
            x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
                polynomial Vandermonde matrix.
            out (ndarray, optional): Shape (`N`, `\\binom{n+d}{d}`, `d`) array to
                hold output (or, if ``C`` is given, shape (`N`, `M`, `d`), where
                `M` is the number of columns of `C`).
            C (ndarray, optional): Shape (`\\binom{n+d}{d}`, `M`) matrix to
                multiply the PDK Vandermonde matrix by on the right.

        Returns:
            ndarray: Shape (`N`, `\\binom{n+d}{d}`, `d`), evaluation of the
            gradient of the PKD Vandermonde matrix at the points `x`.

            Or, if `C` is given, shape (`N`, `M`, `d`), the gradient of the product
            of the Vandermonde matrix with `C`.
        '''
        d = self.reference_element.d
        n = self.reference_element.n
        x = self.reference_element.nodes
        out=None
        C=None
        work=None
        both=False

        N = self._npolys(d, n)
        if out:
            Vg = out
        else:
            if C is None:
                Vg = np.ndarray(x.shape[0:-1]+(N, d))
            else:
                Vg = np.ndarray(x.shape[0:-1]+(C.shape[1], d))
        if both:
            if C is None:
                V = np.ndarray(x.shape[0:-1]+(N,))
            else:
                V = np.ndarray(x.shape[0:-1]+(C.shape[1],))
        if both:
            V[:] = 0.
        Vg[:] = 0.
        if work is None:
            work = np.ndarray(x.shape[0:-1] + (d,))
        for (k, i) in enumerate(self._multiindex_up_to(d, n)):
            if both:
                v, work = self.reference_element.proriolkoornwinderdubinergrad(d, i, x, out=work,
                                                        both=True)
            else:
                work = self.reference_element.proriolkoornwinderdubinergrad(d, i, x, out=work)
            if C is None:
                if both:
                    V[:, k] = v
                    Vg[:, k, 0:d] = work
                else:
                    Vg[:, k, 0:d] = work
            else:
                if both:
                    V[:, :] += np.einsum('i,j->ij', v, C[k, :])
                    Vg[:, :, :] += np.einsum('ij,k->ikj', work, C[k, :])
                else:
                    Vg[:, :, :] += np.einsum('ij,k->ikj', work, C[k, :])
        if both:
            return (V, Vg)
        self.vandermonde_3d_r_derivative = Vg[:,:,0]
        self.vandermonde_3d_s_derivative = Vg[:,:,1]
        self.vandermonde_3d_t_derivative = Vg[:,:,2]
        #return Vg



    def _build_mass_matrix(self):
        """ Build Mass Matrix M"""
        # get inverse vandermonde
        invV = self.inverse_vandermonde_3d

        # compute and store result
        self.mass_matrix = invV.T @ invV


    def _build_differentiation_matrices(self):
        """ Build Differentiation matricies Ds, Dr, and Dt"""
        # get vandermonde matrices
        V = self.vandermonde_3d
        Vr = self.vandermonde_3d_r_derivative 
        Vs = self.vandermonde_3d_s_derivative
        Vt = self.vandermonde_3d_t_derivative

        # compute 
        Dr = np.matmul(Vr, np.linalg.inv(V))
        Ds = np.matmul(Vs, np.linalg.inv(V))
        Dt = np.matmul(Vt, np.linalg.inv(V))

        # store result
        self.r_differentiation_matrix = Dr
        self.s_differentiation_matrix = Ds
        self.t_differentiation_matrix = Dt


    def _build_lift_matrix(self):
        """ Compute 3D surface to volume lift operator used in DG formulation """
        
        # definition of constants
        n = self.reference_element.n
        Np = self.reference_element.nodes_per_cell
        Npf = self.reference_element.nodes_per_face
        num_faces = self.reference_element.num_faces 
        face_node_indices = self.reference_element.face_node_indices
        r = self.reference_element.r
        s = self.reference_element.s
        t = self.reference_element.t
        V = self.vandermonde_3d
        
        # rearrange face_mask
        face_node_indices = face_node_indices.reshape(4, -1).T
        
        # initiate epsilon matrix
        epsilon_matrix = np.zeros((Np, num_faces * Npf))
        
        for face in range(num_faces):
            # get the nodes on the specific face
            if face == 0:
                faceR = r[face_node_indices[:, 0]]
                faceS = s[face_node_indices[:, 0]]
            elif face == 1:
                faceR = r[face_node_indices[:, 1]]
                faceS = t[face_node_indices[:, 1]]
            elif face == 2:
                faceR = s[face_node_indices[:, 2]]
                faceS = t[face_node_indices[:, 2]]
            elif face == 3:
                faceR = s[face_node_indices[:, 3]]
                faceS = t[face_node_indices[:, 3]]
                
            # produce the reference operators on the faces

            face_nodes = np.column_stack((faceR, faceS))
            vandermonde_2d = self._build_vandermonde(2, face_nodes)
            mass_matrix_on_face = np.linalg.inv(vandermonde_2d @ vandermonde_2d.T)
            
            row_index = face_node_indices[:, face]
            column_index = np.arange((face) * Npf, (face + 1) * Npf)
            
            epsilon_matrix[row_index[:, np.newaxis], column_index] += mass_matrix_on_face
                
        self.lift_matrix = V @ (V.T @ epsilon_matrix)
