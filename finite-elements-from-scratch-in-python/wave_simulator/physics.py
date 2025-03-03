import numpy as np
from wave_simulator.mesh import Mesh3d


class LinearAcoustics:
    def __init__(self, mesh: Mesh3d):
        self.mesh = mesh
        num_cells = mesh.num_cells
        num_faces = mesh.reference_element.num_faces
        nodes_per_cell = mesh.reference_element.nodes_per_cell
        nodes_per_face = mesh.reference_element.nodes_per_face
        # define pressure and velocity fields over global nodes
        self.u = np.zeros((nodes_per_cell, num_cells)) # x component of velocity field 
        self.v = np.zeros((nodes_per_cell, num_cells)) # y component of velocity field
        self.w = np.zeros((nodes_per_cell, num_cells)) # z component of velocity field
        self.p = np.zeros((nodes_per_cell, num_cells)) # pressure field 
        # define jumps in fields across faces nodes
        self.du = np.zeros((nodes_per_face * num_faces, num_cells))
        self.dv = np.zeros((nodes_per_face * num_faces, num_cells))
        self.dw = np.zeros((nodes_per_face * num_faces, num_cells))
        self.dp = np.zeros((nodes_per_face * num_faces, num_cells))
        self.alpha = 0.5  # upwinding factor
        self.set_initial_conditions()

    def set_initial_conditions(self, kind="gaussian", center=(0.5, 0.5, 0.5), sigma=0.1, wavelength=1):
        """Set initial conditions for testing the wave propagation."""
        x = self.mesh.x
        y = self.mesh.y
        z = self.mesh.z
        
        if kind == "gaussian":
            # Gaussian pulse centered at (x0, y0, z0)
            x0, y0, z0 = center
            self.p = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
            
            # Normalize so max amplitude is 1
            #self.p /= np.max(self.p)
            
        elif kind == "sine":
            # Plane wave in the x-direction
            k = 2 * np.pi / wavelength
            self.p = np.sin(k * x)


    def compute_rhs(self):
        self._compute_jump_along_normal()
        self._apply_boundary_conditions()
        self._compute_flux()
        return self.rhs_u, self.rhs_v, self.rhs_w, self.rhs_p


    def _compute_jump_along_normal(self):
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices

        # compute jumps at faces
        self.du = (np.ravel(self.u, order='f')[exterior_values] - np.ravel(self.u, order='F')[interior_values])
        self.dv = (np.ravel(self.v, order='F')[exterior_values] - np.ravel(self.v, order='F')[interior_values])
        self.dw = (np.ravel(self.w, order='F')[exterior_values] - np.ravel(self.w, order='F')[interior_values])
        self.dp = (np.ravel(self.p, order='F')[exterior_values] - np.ravel(self.p, order='F')[interior_values])


    def _apply_boundary_conditions(self):
        # Apply reflective conditions: u+ = -u-, p+ = p-
        boundary_face_node_indices = self.mesh.boundary_face_node_indices
        boundary_node_indices = self.mesh.boundary_node_indices
        # Normal velocity reverses sign: u+ = -u-
        self.du[boundary_face_node_indices] = -2 * np.ravel(self.u, order='F')[boundary_node_indices]
        self.dv[boundary_face_node_indices] = -2 * np.ravel(self.v, order='F')[boundary_node_indices]
        self.dw[boundary_face_node_indices] = -2 * np.ravel(self.w, order='F')[boundary_node_indices]
        # Pressure remains unchanged: p+ = p-
        self.dp[boundary_face_node_indices] = 0


    def _compute_flux(self):
        # spatial derivative matrices
        Dr = self.mesh.reference_element_operators.r_differentiation_matrix
        Ds = self.mesh.reference_element_operators.s_differentiation_matrix
        Dt = self.mesh.reference_element_operators.t_differentiation_matrix
        face_scale = self.mesh.surface_to_volume_jacobian
        lift = self.mesh.reference_element_operators.lift_matrix

        # local spatial derivatives on reference tetrahedron
        drdx = self.mesh.drdx
        drdy = self.mesh.drdy
        drdz = self.mesh.drdz
        dsdx = self.mesh.dsdx
        dsdy = self.mesh.dsdy
        dsdz = self.mesh.dsdz
        dtdx = self.mesh.dtdx
        dtdy = self.mesh.dtdy
        dtdz = self.mesh.dtdz

        # compute derivatives in physical space
        dudx = drdx * (Dr @ self.u) + dsdx * (Ds @ self.u) + dtdx * (Dt @ self.u)
        dvdy = drdy * (Dr @ self.v) + dsdy * (Ds @ self.v) + dtdy * (Dt @ self.v)
        dwdz = drdz * (Dr @ self.w) + dsdz * (Ds @ self.w) + dtdz * (Dt @ self.w)
        dpdx = drdx * (Dr @ self.p) + dsdx * (Ds @ self.p) + dtdx * (Dt @ self.p)
        dpdy = drdy * (Dr @ self.p) + dsdy * (Ds @ self.p) + dtdy * (Dt @ self.p)
        dpdz = drdz * (Dr @ self.p) + dsdz * (Ds @ self.p) + dtdz * (Dt @ self.p)

        # reshape jump matrices
        Npf = self.mesh.reference_element.nodes_per_face
        num_faces = self.mesh.reference_element.num_faces
        K = self.mesh.num_cells
        self.du = self.du.reshape((Npf*num_faces, K), order='F')
        self.dv = self.dv.reshape((Npf*num_faces, K), order='F')
        self.dw = self.dw.reshape((Npf*num_faces, K), order='F')
        self.dp = self.dp.reshape((Npf*num_faces, K), order='F')
        
        # Compute normal jump of pressure
        ndotdu = self.mesh.nx * self.du + self.mesh.ny * self.dv + self.mesh.nz * self.dw
        ndotdp = self.mesh.nx * self.dp + self.mesh.ny * self.dp + self.mesh.nz * self.dp
        
        # Compute fluxes 
        flux_u = -self.mesh.nx * (self.alpha * ndotdu - self.dp)
        flux_v = -self.mesh.ny * (self.alpha * ndotdu - self.dp)
        flux_w = -self.mesh.nz * (self.alpha * ndotdu - self.dp)
        flux_p = (ndotdu - self.alpha * self.dp)

        # Compute right-hand side terms using lifting operation
        rhs_u = -dpdx + lift @ (face_scale * flux_u / 2)
        rhs_v = -dpdy + lift @ (face_scale * flux_v / 2)
        rhs_w = -dpdz + lift @ (face_scale * flux_w / 2)
        rhs_p = -(dudx + dvdy + dwdz) + lift @ (face_scale * flux_p / 2)

        # Store results
        self.rhs_u = rhs_u
        self.rhs_v = rhs_v
        self.rhs_w = rhs_w
        self.rhs_p = rhs_p

        self.rhs_u = rhs_u


if __name__ == "__main__":
    import sys
    from wave_simulator.finite_elements import LagrangeElement
    from wave_simulator.visualizing import *

    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        mesh_file = "./inputs/meshes/simple.msh"
    
    dim = 3
    n = 3
    mesh = mesh3d(mesh_file, LagrangeElement(dim,n))
    Physics = LinearAcoustics(mesh)
    Physics.compute_rhs()
    breakpoint()

    pass





