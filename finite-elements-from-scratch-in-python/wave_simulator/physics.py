import numpy as np
from wave_simulator.mesh import Mesh3d


class LinearAcoustics:
    def __init__(self, mesh: Mesh3d):
        self.mesh = mesh
        num_cells = mesh.num_cells
        #num_faces = mesh.reference_element.num_faces
        nodes_per_cell = mesh.reference_element.nodes_per_cell
        #nodes_per_face = mesh.reference_element.nodes_per_face
        # define pressure and velocity fields over global nodes
        #self.u = np.zeros((nodes_per_cell, num_cells)) # x component of velocity field 
        #self.v = np.zeros((nodes_per_cell, num_cells)) # y component of velocity field
        #self.w = np.zeros((nodes_per_cell, num_cells)) # z component of velocity field
        #self.p = np.zeros((nodes_per_cell, num_cells)) # pressure field 
        #self.density = np.ones((nodes_per_cell, num_cells)) * 1.2 # kg/m^3 density at each node
        #self.speed = np.ones((nodes_per_cell, num_cells)) * 343 # m/s wavespeed at each node
        ## define jumps in fields across faces nodes
        #self.du = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dv = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dw = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dp = np.zeros((nodes_per_face * num_faces, num_cells))
        self.density = np.ones((nodes_per_cell, num_cells)) * 1 # kg/m^3 density at each node
        self.speed = np.ones((nodes_per_cell, num_cells)) * 1 # m/s wavespeed at each node
        self.speed[(np.abs(self.mesh.x -0.5) < 0.2) & (np.abs(self.mesh.y - 0.5) < 0.2) & (np.abs(self.mesh.z -0.5) < 0.2)] = 10
        #self.speed[self.mesh.z < 0.3] = 1
        # Calculate the mean of each column
        average_cell_speed = np.mean(self.speed, axis=0)
        # Replace each column with its mean value
        self.speed = np.tile(average_cell_speed, (self.speed.shape[0], 1))
        self.density = self.speed
        self.max_speed = np.max(self.speed)
        self.set_initial_conditions()

    def set_initial_conditions(self, kind="zero", center=(0.80, 0.50, 0.50), sigma=0.05, wavelength=1):
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
            self.p = 10*np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
        elif kind == "mode":
            self.p = np.sin(2*np.pi * x) + np.sin(2*np.pi *y) + np.sin(2*np.pi *z)
            self.p[x**2 + y**2 + z**2 > 1] = 0


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


    def _get_material_face_values(self):
        # indices for interior and exterior values
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices
        # get interior values on cells
        rho_m = self.density.ravel(order='F')[interior_values]
        rho_p = self.density.ravel(order='F')[interior_values]
        c_p = self.speed.ravel(order='F')[exterior_values]
        c_m = self.speed.ravel(order='F')[exterior_values]
        rho_p, rho_m, c_p, c_m = self._reshape_to_rectangular(rho_p, rho_m, c_p, c_m)
        return rho_p, rho_m, c_p, c_m


    def _apply_boundary_conditions(self):
        # indices for interior and exterior values
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices

        # indices for face values
        boundary = self.mesh.boundary_node_indices
        face_boundary = self.mesh.boundary_face_node_indices

        # get interior values on cells
        u_m = self.u.ravel(order='F')[interior_values]
        v_m = self.v.ravel(order='F')[interior_values]
        w_m = self.w.ravel(order='F')[interior_values]
        p_m = self.p.ravel(order='F')[interior_values]

        # get exterior values (just outside of each cell)
        u_p = self.u.ravel(order='F')[exterior_values]
        v_p = self.v.ravel(order='F')[exterior_values]
        w_p = self.w.ravel(order='F')[exterior_values]
        p_p = self.p.ravel(order='F')[exterior_values]

        #  ravel normal vectors for indexing
        nx = self.mesh.nx.ravel(order='F')
        ny = self.mesh.ny.ravel(order='F')
        nz = self.mesh.nz.ravel(order='F')

        # compute normal velocity on interior boundary cells
        ndotum = nx[face_boundary]* u_m[face_boundary] + ny[face_boundary] * v_m[face_boundary] + nz[face_boundary] * w_m[face_boundary]

        # calculate velocity and pressure on external boundary cells
        u_p[face_boundary] = u_m[face_boundary] - 2.0 * (ndotum) * nx[face_boundary]
        v_p[face_boundary] = v_m[face_boundary] - 2.0 * (ndotum) * ny[face_boundary]
        w_p[face_boundary] = w_m[face_boundary] - 2.0 * (ndotum) * nz[face_boundary]
        p_p[face_boundary] = p_m[face_boundary]

        # reshape for matrix-matrix multiplication
        u_m, v_m, w_m, p_m = self._reshape_to_rectangular(u_m, v_m, w_m, p_m)
        u_p, v_p, w_p, p_p = self._reshape_to_rectangular(u_p, v_p, w_p, p_p)
        return u_p, v_p, w_p, p_p, u_m, v_m, w_m, p_m 


    def compute_rhs(self, u=None, v=None, w=None, p=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if p is None:
            p = self.p

        # get material parameters 
        rho = self.density
        c = self.speed

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

        # apply boundary conditions
        u_p, v_p, w_p, p_p, u_m, v_m, w_m, p_m = self._apply_boundary_conditions()

        # get heterogeneous material matrices
        rho_p, rho_m, c_p, c_m = self._get_material_face_values()

        # compute normal velocity at interior boundary and exterior boundary 
        ndotum = self.mesh.nx * u_m + self.mesh.ny * v_m + self.mesh.nz * w_m
        ndotup = self.mesh.nx * u_p + self.mesh.ny * v_p + self.mesh.nz * w_p

        # compute fluxes 
        # indices for interior and exterior values
        interior = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices
        num_faces = self.mesh.reference_element.num_faces
        Npf = self.mesh.reference_element.nodes_per_face

        #flux_u = 0.5 * (self.mesh.nx * ((p_p - p_m) - (ndotup - ndotum)))
        #flux_v = 0.5 * (self.mesh.ny * ((p_p - p_m) - (ndotup - ndotum)))
        #flux_w = 0.5 * (self.mesh.nz * ((p_p - p_m) - (ndotup - ndotum)))
        #flux_p = 0.5 * ((ndotup - ndotum) - (p_p - p_m))
        mu = np.maximum(c_p, c_m)

        flux_p = 0.5 * (
            (rho_p * c_p**2 * u_p - rho_m * c_m**2 * u_m) * self.mesh.nx + \
            (rho_p * c_p**2 * v_p - rho_m * c_m**2 * v_m) * self.mesh.ny + \
            (rho_p * c_p**2 * w_p - rho_m * c_m**2 * w_m) * self.mesh.nz - \
            mu * (p_p - p_m)
        )
        flux_u = 0.5 * (self.mesh.nx * ((p_p / rho_p) - (p_m / rho_m) - mu * (ndotup - ndotum)))
        flux_v = 0.5 * (self.mesh.ny * ((p_p / rho_p) - (p_m / rho_m) - mu * (ndotup - ndotum)))
        flux_w = 0.5 * (self.mesh.nz * ((p_p / rho_p) - (p_m / rho_m) - mu * (ndotup - ndotum)))

        # compute right-hand side terms using lifting operation
        self.rhs_u = -dpdx - lift @ (face_scale * flux_u)
        self.rhs_v = -dpdy - lift @ (face_scale * flux_v)
        self.rhs_w = -dpdz - lift @ (face_scale * flux_w)
        self.rhs_p = -(dudx + dvdy + dwdz) - lift @ (face_scale * flux_p)

        return self.rhs_u, self.rhs_v, self.rhs_w, self.rhs_p,
