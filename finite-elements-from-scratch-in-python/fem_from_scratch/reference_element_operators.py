import numpy as np

class ReferenceElementOperators:

    def __init__(self, FiniteElement):
        self.ReferenceElement =  FiniteElement
        Np = FiniteElement.nodes_per_element
        Npf = FiniteElement.nodes_per_face
        r = self.ReferenceElement.r
        s = self.ReferenceElement.s
        t = self.ReferenceElement.t
        self.vandermonde_2d = np.zeros((len(r), Npf))
        self.vandermonde_3d = np.zeros((len(r), Np))
        self.vandermonde_3d_r_derivative = np.zeros((len(r), Np))
        self.vandermonde_3d_s_derivative = np.zeros((len(r), Np))
        self.vandermonde_3d_t_derivative = np.zeros((len(r), Np))
        self.inverse_vandermonde_3d = np.zeros((Np,Np))
        self.mass_matrix = np.zeros((Np,Np))
        self.s_differentiation_matrix = np.zeros((Np, Np))
        self.r_differentiation_matrix = np.zeros((Np, Np))
        self.t_differentiation_matrix = np.zeros((Np, Np))

        self._calculate_element_operators(r,s,t)
        

    def _calculate_element_operators(self, r, s, t):
        self._build_vandermonde_2d(r,s)
        self._build_vandermonde_3d(r,s,t)
        self._build_inverse_vandermonde_3d()
        self._build_vandermonde_3d_gradient(r,s,t)
        self._build_mass_matrix()
        self._build_differentiation_matrices()
        self._build_lift_matrix()
        
        
    def _build_vandermonde_3d(self, r, s, t):
        """ create 3D vandermonde matrix"""

        # initialize the 3D Vandermonde Matrix
        V = self.vandermonde_3d
        # get orthonormal basis
        eval_basis_function_3d = self.ReferenceElement.eval_basis_function_3d
        # get polynomial order of finite element
        n = self.ReferenceElement.n 
        
        # build the Vandermonde matrix
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    V[:, column_index] = eval_basis_function_3d(r, s, t, i, j, k)
                    column_index += 1

        # store result
        self.vandermonde_3d = V


    def _build_vandermonde_2d(self, r, s):
        """ create 2D vandermonde matrix to evaluate flux at faces of each element"""
        
        # initiate vandermonde matrix
        V = self.vandermonde_2d

        V = np.zeros((len(r), self.ReferenceElement.nodes_per_face))
        
        # get basis function
        eval_basis_function_2d = self.ReferenceElement.eval_basis_function_2d

        # get polynomial order of finite element
        n = self.ReferenceElement.n 

        # build the Vandermonde matrix
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                V[:, column_index] = eval_basis_function_2d(r, s, i, j)
                column_index += 1

        # store result
        self.vandermonde_2d = V
                

    def _build_inverse_vandermonde_3d(self):
        """ invert the 3D Vandermonde Matrix """
        self.inverse_vandermonde_3d = np.linalg.inv(self.vandermonde_3d)


    def _build_vandermonde_3d_gradient(self, r, s, t):
        """ Build gradient (Vr, Vs, Vt) of Vandermonde matrix"""
        # initialize Vandermonde derivative matrices
        Vr = self.vandermonde_3d_r_derivative
        Vs = self.vandermonde_3d_s_derivative
        Vt = self.vandermonde_3d_t_derivative
        
        # get basis function
        eval_basis_function_3d_gradient = self.ReferenceElement.eval_basis_function_3d_gradient

        # get polynomial order of finite element
        n = self.ReferenceElement.n 
        
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
        pass


