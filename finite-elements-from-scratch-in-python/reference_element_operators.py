from finite_elements import LagrangeElement

class referenceElementOperators:

    def __init__(self, finiteElement):
        self.finiteElement =  finiteElement
        Np = finiteElement.nodesPerElement
        Npf = finiteElement.nodesPerFace
        self.vandermonde_3d = np.zeros((Np, Np))
        self.vandermonde_2d = np.zeros((Npf, Npf))

        self._calculate_element_operators()
        

    def _calculate_element_operators(self):
        r = self.finiteElement.r
        s = self.finiteElement.s
        t = self.finiteElement.t
        self._build_vandermonde_3d(r,s,t)
        self._build_vandermonde_2d(r,s)
        
        
    def _build_vandermonde_3d(self, r, s, t):
        """ create 3D vandermonde matrix"""

        # Initialize the 3D Vandermonde Matrix
        V = self.vandermonde_3d
        # get orthonormal basis
        orthonormal_basis = self.finiteElement.eval_basis_function()
        # get polynomial order of finite element
        n = self.finiteElement.n 
        
        # Build the Vandermonde matrix
        column_index = 0
        for i in range(n + 1):
            for j in range(n - i + 1):
                for k in range(n - i - j + 1):
                    V[:, column_index] = orthonormal_basis(r, s, t, i, j, k)
                    column_index += 1
        self.vandermonde_3d = V


    def _build_vandermonde_2d(r, s):
        """ create 2D vandermonde matrix to evaluate flux at faces of each element"""
        
        # initiate vandermonde matrix
        V = self.vandermonde_2d
        
        # get basis function
        orthonormal_basis_2d = self.finiteElement.eval_basis_function_2d()

        # get polynomial order of finite element
        n = self.finiteElement.n 

        # Build the Vandermonde matrix
        column_index = 0
        for i in range(n+1):
            for j in range(n - i + 1):
                v[:, column_index] = orthonormal_basis_2d(r, s, i, j)
                column_index += 1

        self.vandermonde_2d = V
                
