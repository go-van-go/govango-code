class ReferenceElementOperators:

    def __init__(self, FiniteElement):
        self.ReferenceElement =  FiniteElement
        Np = FiniteElement.nodes_per_element
        Npf = FiniteElement.nodes_per_face
        self.vandermonde_3d = np.zeros((Np, Np))
        self.vandermonde_2d = np.zeros((Npf, Npf))
        self.vandermonde_3d_r_derivative = np.zeros((Np, Np))
        self.vandermonde_3d_s_derivative = np.zeros((Np, Np))
        self.vandermonde_3d_t_derivative = np.zeros((Np, Np))
        self.inverse_vandermonde_3d = np.zeros((Np,Np))
        self.mass_matrix = np.zeros((Np,Np))

        self._calculate_element_operators()
        

    def _calculate_element_operators(self):
        r = self.ReferenceElement.r
        s = self.ReferenceElement.s
        t = self.ReferenceElement.t
        self._build_vandermonde_3d(r,s,t)
        self._build_vandermonde_2d(r,s)
        self._build_inverse_vandermonde_3d()
        self._build_vandermonde_3d_gradient(r,s,t)
        self._build_massMatrix()
        
        
    def _build_vandermonde_3d(self, r, s, t):
        """ create 3D vandermonde matrix"""

        # Initialize the 3D Vandermonde Matrix
        V = self.vandermonde_3d
        # get orthonormal basis
        eval_basis_function_3d = self.ReferenceElement.eval_basis_function_3d()
        # get polynomial order of finite element
        n = self.ReferenceElement.n 
        
        # Build the Vandermonde matrix
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    V[:, column_index] = eval_basis_function_3d(r, s, t, i, j, k)
                    column_index += 1

        self.vandermonde_3d = V


    def _build_vandermonde_2d(r, s):
        """ create 2D vandermonde matrix to evaluate flux at faces of each element"""
        
        # initiate vandermonde matrix
        V = self.vandermonde_2d
        
        # get basis function
        eval_basis_function_2d = self.finiteElement.eval_basis_function_2d()

        # get polynomial order of finite element
        n = self.ReferenceElement.n 

        # Build the Vandermonde matrix
        column_index = 0
        for i in range(n+1):
            for j in range(n - i + 1):
                v[:, column_index] = eval_basis_function_2d(r, s, i, j)
                column_index += 1

        self.vandermonde_2d = V
                

    def _build_inverse_vandermonde_3d(self):
        self.inverse_vandermonde_3d = np.linalg.inv(self.vandermonde_3d)


    def _build_vandermonde_3d_gradient(r, s, t):
        # initialize Vandermonde derivative matrices
        Vr = self.vandermonde_3d_r_derivative
        Vs = self.vandermonde_3d_s_derivative
        Vt = self.vandermonde_3d_t_derivative
        
        # get basis function
        eval_basis_function_3d_gradient = self.finiteElement.eval_basis_function_3d_gradient()

        # get polynomial order of finite element
        n = self.ReferenceElement.n 
        
        # Build Vandermonde derivative matrices
        column_index = 0
        for i in range(n + 1):
            for j in range(N - i + 1):
                for k in range(N - i - j + 1):
                    Vr[:, column_index], Vs[:, column_index], Vt[:, column_index] = \
                        eval_basis_function_3d_gradient(r, s, t, i, j, k)
                    column_index += 1
                    
        self.vandermonde_3d_r_derivative = Vr
        self.vandermonde_3d_s_derivative = Vs
        self.vandermonde_3d_t_derivative = Vt
 

    def _build_massMatrix(self):
        invV = self.inverse_vandermonde_3d
        self.mass_matrix = invV.T @ invV
