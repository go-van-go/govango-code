import numpy as np
from wave_simulator.visualizing import *
from wave_simulator.finite_elements import LagrangeElement

class ReferenceElementOperators:

    def __init__(self, finite_element: LagrangeElement):
        self.reference_element =  finite_element
        nodes_per_cell = finite_element.nodes_per_cell
        nodes_per_face = finite_element.nodes_per_face
        num_faces = finite_element.num_faces
        r = self.reference_element.r
        self.vandermonde_2d = np.zeros((nodes_per_face, nodes_per_face))
        self.vandermonde_3d = np.zeros((len(r), nodes_per_cell))
        self.vandermonde_3d_r_derivative = np.zeros((len(r), nodes_per_cell))
        self.vandermonde_3d_s_derivative = np.zeros((len(r), nodes_per_cell))
        self.vandermonde_3d_t_derivative = np.zeros((len(r), nodes_per_cell))
        self.inverse_vandermonde_3d = np.zeros((nodes_per_cell,nodes_per_cell))
        self.mass_matrix = np.zeros((nodes_per_cell,nodes_per_cell))
        self.s_differentiation_matrix = np.zeros((nodes_per_cell, nodes_per_cell))
        self.r_differentiation_matrix = np.zeros((nodes_per_cell, nodes_per_cell))
        self.t_differentiation_matrix = np.zeros((nodes_per_cell, nodes_per_cell))
        self.lift_matrix = np.zeros((nodes_per_cell, num_faces * nodes_per_face))
        

        self._calculate_element_operators()
        

    def _calculate_element_operators(self):
        r = self.reference_element.r
        s = self.reference_element.s
        t = self.reference_element.t
        self._build_vandermonde_3d(r,s,t)
        self._build_inverse_vandermonde_3d()
        self._build_vandermonde_3d_gradient(r,s,t)
        self._build_mass_matrix()
        self._build_differentiation_matrices()
        self._build_lift_matrix()
        
        
    def _build_vandermonde_3d(self, r, s, t):
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


    def _build_vandermonde_2d(self, r, s):
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


    def _build_vandermonde_3d_gradient(self, r, s, t):
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
            vandermonde_2d = self._build_vandermonde_2d(faceR, faceS)
            mass_matrix_on_face = np.linalg.inv(vandermonde_2d @ vandermonde_2d.T)
            
            row_index = face_node_indices[:, face]
            column_index = np.arange((face) * Npf, (face + 1) * Npf)
            
            epsilon_matrix[row_index[:, np.newaxis], column_index] += mass_matrix_on_face
                
        self.lift_matrix = V @ (V.T @ epsilon_matrix)


