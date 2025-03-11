import numpy as np
from wave_simulator.mesh import Mesh3d


class LinearAcoustics:
    def __init__(self, mesh: Mesh3d):
        self.mesh = mesh
        #num_cells = mesh.num_cells
        #num_faces = mesh.reference_element.num_faces
        #nodes_per_cell = mesh.reference_element.nodes_per_cell
        #nodes_per_face = mesh.reference_element.nodes_per_face
        # define pressure and velocity fields over global nodes
        #self.u = np.zeros((nodes_per_cell, num_cells)) # x component of velocity field 
        #self.v = np.zeros((nodes_per_cell, num_cells)) # y component of velocity field
        #self.w = np.zeros((nodes_per_cell, num_cells)) # z component of velocity field
        #self.p = np.zeros((nodes_per_cell, num_cells)) # pressure field 
        ## define jumps in fields across faces nodes
        #self.du = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dv = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dw = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dp = np.zeros((nodes_per_face * num_faces, num_cells))
        self.alpha = 1  # upwinding factor
        self.density = 1.2 # kg/m^3
        self.speed = 343 # m/s
        self.set_initial_conditions()

    def set_initial_conditions(self, kind="gaussian", center=(0.5, 0.5, 0.5), sigma=0.08, wavelength=1):
        """Set initial conditions for testing the wave propagation."""
        x = self.mesh.x
        y = self.mesh.y
        z = self.mesh.z
        
        num_cells = self.mesh.num_cells
        nodes_per_cell = self.mesh.reference_element.nodes_per_cell
 
        self.u = np.zeros((nodes_per_cell,  num_cells)) # v_x field 
        self.v = np.zeros((nodes_per_cell,  num_cells)) # v_y field 
        self.w = np.zeros((nodes_per_cell,  num_cells)) # v_z field 
        self.p = np.zeros((nodes_per_cell,  num_cells)) # pressure field 

        if kind == "gaussian":
            # Gaussian pulse centered at (x0, y0, z0)
            x0, y0, z0 = center
            # define pressure and velocity fields over global nodes
            self.p = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))

        elif kind == "wall":
            # pressure plane wave in the x-direction
            x0 = 0.5
            y0 = 0.5
            z0 = 0.5
            width = 0.01
            #x[abs(x-x0) > width] = 0
            #y[abs(y-y0) > width] = 0
            #z[abs(z-z0) > width] = 0
            #z[abs(x-z0) > width] = 0
            #z[abs(y-z0) > width] = 0
            #self.u[abs(x-x0) < width] = 0.1
            #self.p[(abs(x - x0) < width) & (abs(y - y0) < 0.3) & (abs(z-z0) < 0.3)] = 0.1
            self.u[(np.abs(x-x0) < width) & (np.abs(y - y0) < 0.2) & (np.abs(z - z0) < 0.2)]=-1
            #self.u = self.u*np.exp(-(x-1)*10) 
            #self.u = self.u*np.exp(-(y-0.5)**2) 
            #self.u = self.u*np.exp(-(z-0.5)**2) 

        elif kind == "mode":
            self.p = np.sin(2*np.pi * x) + np.sin(2*np.pi *y) + np.sin(2*np.pi *z)
            self.p[x**2 + y**2 + z**2 > 1] = 0


    def compute_rhs(self):
        #self._compute_jump_along_normal()
        #self._apply_boundary_conditions()
        self._compute_flux()
        return self.rhs_u, self.rhs_v, self.rhs_w, self.rhs_p

    def _reshape_to_rectangular(self, du, dv, dw, dp):
        # reshape jump matrices
        Npf = self.mesh.reference_element.nodes_per_face
        num_faces = self.mesh.reference_element.num_faces
        K = self.mesh.num_cells
        du = du.reshape((Npf*num_faces, K), order='F')
        dv = dv.reshape((Npf*num_faces, K), order='F')
        dw = dw.reshape((Npf*num_faces, K), order='F')
        dp = dp.reshape((Npf*num_faces, K), order='F')
        return du, dv, dw, dp
 

    def _compute_positive_jump_along_normal(self, u, v, w, p):
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices

        # compute jumps at faces
        du_p = u.ravel(order='F')[exterior_values] + u.ravel(order='F')[interior_values]
        dv_p = v.ravel(order='F')[exterior_values] + v.ravel(order='F')[interior_values]
        dw_p = w.ravel(order='F')[exterior_values] + w.ravel(order='F')[interior_values]
        dp_p = p.ravel(order='F')[exterior_values] + p.ravel(order='F')[interior_values]

        du_p, dv_p, dw_p, dp_p = self._reshape_to_rectangular(du_p, dv_p, dw_p, dp_p)
        return du_p, dv_p, dw_p, dp_p


    def _compute_negative_jump_along_normal(self, u, v, w, p):
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices

        # compute jumps at faces
        du_m = u.ravel(order='F')[interior_values] - u.ravel(order='F')[exterior_values]
        dv_m = v.ravel(order='F')[interior_values] - v.ravel(order='F')[exterior_values]
        dw_m = w.ravel(order='F')[interior_values] - w.ravel(order='F')[exterior_values]
        dp_m = p.ravel(order='F')[interior_values] - p.ravel(order='F')[exterior_values]

        du_m, dv_m, dw_m, dp_m = self._reshape_to_rectangular(du_m, dv_m, dw_m, dp_m)
        return du_m, dv_m, dw_m, dp_m



    def old_apply_boundary_conditions(self):
        # Apply reflective conditions: u+ = -u-, p+ = p-
        boundary_face_node_indices = self.mesh.boundary_face_node_indices
        boundary_node_indices = self.mesh.boundary_node_indices

        ndotdu = self.mesh.nx.ravel(order='F')[boundary_face_node_indices] * self.u.ravel(order='F')[boundary_node_indices] + \
        self.mesh.ny.ravel(order='F')[boundary_face_node_indices] * self.v.ravel(order='F')[boundary_node_indices] + \
        self.mesh.nz.ravel(order='F')[boundary_face_node_indices] * self.w.ravel(order='F')[boundary_node_indices]
        # Normal velocity reverses sign: u+ = -u-
        self.du[boundary_face_node_indices] = self.u.ravel(order='F')[boundary_node_indices] - \
            2*self.u.ravel(order='F')[boundary_node_indices]*self.mesh.nx.ravel(order='F')[boundary_face_node_indices]
        self.dv[boundary_face_node_indices] = self.v.ravel(order='F')[boundary_node_indices] - \
            2*self.v.ravel(order='F')[boundary_node_indices]*self.mesh.ny.ravel(order='F')[boundary_face_node_indices]
        self.dw[boundary_face_node_indices] = self.w.ravel(order='F')[boundary_node_indices] - \
            2*self.w.ravel(order='F')[boundary_node_indices]*self.mesh.nz.ravel(order='F')[boundary_face_node_indices]

        # Pressure remains unchanged: p+ = p-
        self.dp[boundary_face_node_indices] = 0


    def _apply_boundary_conditions(self, u_p, v_p, w_p, p_p, u_m, v_m, w_m, p_m):
        face_boundary = self.mesh.boundary_face_node_indices
        boundary = self.mesh.boundary_node_indices
        nx = self.mesh.nx.ravel(order='F')
        ny = self.mesh.ny.ravel(order='F')
        nz = self.mesh.nz.ravel(order='F')

        ndotum = nx[face_boundary]* u_m[face_boundary] + ny[face_boundary] * v_m[face_boundary] + nz[face_boundary] * w_m[face_boundary]

        u_p[face_boundary] = u_m[face_boundary] - 2.0 * (ndotum) * nx[face_boundary]
        v_p[face_boundary] = v_m[face_boundary] - 2.0 * (ndotum) * ny[face_boundary]
        w_p[face_boundary] = w_m[face_boundary] - 2.0 * (ndotum) * nz[face_boundary]
        p_p[face_boundary] = p_m[face_boundary]

        return u_p, v_p, w_p, p_p


    def _compute_flux(self):
        # spatial derivative matrices
        Dr = self.mesh.reference_element_operators.r_differentiation_matrix
        Ds = self.mesh.reference_element_operators.s_differentiation_matrix
        Dt = self.mesh.reference_element_operators.t_differentiation_matrix
        face_scale = self.mesh.surface_to_volume_jacobian
        lift = self.mesh.reference_element_operators.lift_matrix

        # indices for interior and exterior values
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices

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

        # Compute normal jump of pressure
        du_p, dv_p, dw_p, dp_p = self._compute_positive_jump_along_normal(self.u, self.v, self.w, self.p)
        du_m, dv_m, dw_m, dp_m = self._compute_negative_jump_along_normal(self.u, self.v, self.w, self.p)

        u_m = self.u.ravel(order='F')[interior_values]
        v_m = self.v.ravel(order='F')[interior_values]
        w_m = self.w.ravel(order='F')[interior_values]
        p_m = self.p.ravel(order='F')[interior_values]

        u_p = self.u.ravel(order='F')[exterior_values]
        v_p = self.v.ravel(order='F')[exterior_values]
        w_p = self.w.ravel(order='F')[exterior_values]
        p_p = self.p.ravel(order='F')[exterior_values]

        u_p, v_p, w_p, p_p = self._apply_boundary_conditions(u_p, v_p, w_p, p_p, u_m, v_m, w_m, p_m)

        u_m, v_m, w_m, p_m = self._reshape_to_rectangular(u_m, v_m, w_m, p_m)
        u_p, v_p, w_p, p_p = self._reshape_to_rectangular(u_p, v_p, w_p, p_p)


        ndotum = self.mesh.nx * u_m + self.mesh.ny * v_m + self.mesh.nz * w_m
        ndotup = self.mesh.nx * u_p + self.mesh.ny * v_p + self.mesh.nz * w_p

        # Compute fluxes 
        flux_u = 0.5 * (self.mesh.nx * (p_p - p_m - (ndotup - ndotum)))
        flux_v = 0.5 * (self.mesh.ny * (p_p - p_m - (ndotup - ndotum)))
        flux_w = 0.5 * (self.mesh.nz * (p_p - p_m - (ndotup - ndotum)))
        flux_p = 0.5 * (ndotup - ndotum - (p_p - p_m))

        # Compute right-hand side terms using lifting operation
        self.rhs_u = -dpdx - lift @ (face_scale * flux_u)
        self.rhs_v = -dpdy - lift @ (face_scale * flux_v)
        self.rhs_w = -dpdz - lift @ (face_scale * flux_w)
        self.rhs_p = -(dudx + dvdy + dwdz) - lift @ (face_scale * flux_p)
