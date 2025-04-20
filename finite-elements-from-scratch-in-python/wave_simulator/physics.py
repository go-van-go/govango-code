import numpy as np
from wave_simulator.mesh import Mesh3d
from wave_simulator.visualizer import Visualizer 
from scipy.stats import mode

class LinearAcoustics:
    def __init__(self, mesh: Mesh3d):
        self.mesh = mesh
        #self.u = np.zeros((nodes_per_cell, num_cells)) # x component of velocity field 
        #self.v = np.zeros((nodes_per_cell, num_cells)) # y component of velocity field
        #self.w = np.zeros((nodes_per_cell, num_cells)) # z component of velocity field
        #self.p = np.zeros((nodes_per_cell, num_cells)) # pressure field 
        ## define jumps in fields across faces nodes
        #self.du = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dv = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dw = np.zeros((nodes_per_face * num_faces, num_cells))
        #self.dp = np.zeros((nodes_per_face * num_faces, num_cells))
        self.source_center = np.array([0.125, 0.125, 0.0])
        self.source_radius = 0.02
        self.source_frequency = 10000  # Hz
        self.source_duration = (1 / self.source_frequency)
        self.source_amplitude = 100000
        # air density = 1.293 earthdata.nasa.gov/topics/atmosphere/air-mass-density
        # air speed = 343
        #self.surface_impedance = 1.293 * (343**2) 
        # totally reflecting
        self.surface_impedance = 0
        self.max_speed = np.max(self.mesh.speed)
        self.set_initial_conditions()

    def set_initial_conditions(self, kind="gaussian"):
        """Set initial conditions for testing the wave propagation."""
        # initialize zero value velocity and pressure fields
        num_cells = self.mesh.num_cells
        nodes_per_cell = self.mesh.reference_element.nodes_per_cell
        self.u = np.zeros((nodes_per_cell,  num_cells)) # v_x field 
        self.v = np.zeros((nodes_per_cell,  num_cells)) # v_y field 
        self.w = np.zeros((nodes_per_cell,  num_cells)) # v_z field 
        self.p = np.zeros((nodes_per_cell,  num_cells)) # pressure field 

        # get vertex coordinates
        x = self.mesh.x
        y = self.mesh.y
        z = self.mesh.z

        # set fields
        if kind == "gaussian":
            # Gaussian pulse centered at (x0, y0, z0)
            #center=(0.1250, 0.1250, 0.1250)
            center=(0.50, 0.50, 0.50)
            sigma=0.05
            x0, y0, z0 = center
            # define pressure field to be a gaussian pulse 
            self.p = 10*np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))

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
        exterior_values = self.mesh.exterior_face_node_indices
        interior_values = self.mesh.interior_face_node_indices
        # get interior values on cells
        rho_p = self.mesh.density.ravel(order='F')[exterior_values]
        c_p = self.mesh.speed.ravel(order='F')[exterior_values]
        rho_m = self.mesh.density.ravel(order='F')[interior_values]
        c_m = self.mesh.speed.ravel(order='F')[interior_values]
        rho_p, rho_m, c_p, c_m = self._reshape_to_rectangular(rho_p, rho_m, c_p, c_m)
        return rho_p, rho_m, c_p, c_m

    def _get_amplitude(self, time):
        # Gaussian envelope
        t0 = 0.5 * self.source_duration  # Center the Gaussian in the middle of the pulse duration
        sigma = self.source_duration / 6  # Controls pulse width
        envelope = np.exp(-((time - t0) ** 2) / (2 * sigma ** 2))
        amplitude = self.source_amplitude * envelope
        return amplitude

    def _apply_source_boundary_condition(self, time, p_p):
        omega = 2 * np.pi * self.source_frequency
        amplitude = self._get_amplitude(time)
        source_pressure = amplitude * np.sin(omega * time)
        
        exterior_values = self.mesh.exterior_face_node_indices
        boundary = self.mesh.boundary_face_node_indices

        # Get global boundary face node coordinates
        x_b = self.mesh.x.ravel(order='F')[exterior_values][boundary]
        y_b = self.mesh.y.ravel(order='F')[exterior_values][boundary]
        z_b = self.mesh.z.ravel(order='F')[exterior_values][boundary]
            
        # Find nodes within circular source region
        in_source = ((x_b - self.source_center[0])**2 +
                     (y_b - self.source_center[1])**2 < self.source_radius**2) & \
                     (np.abs(z_b - self.source_center[2]) < 1e-6)
            
        # Overwrite the pressure with sinusoidal source
        p_p[boundary[in_source]] = source_pressure
        return p_p

 
    def _apply_boundary_conditions(self, time):
        # indices for interior and exterior values
        interior_values = self.mesh.interior_face_node_indices
        exterior_values = self.mesh.exterior_face_node_indices

        # indices for face values
        boundary = self.mesh.boundary_face_node_indices

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
        ndotum = nx[boundary]* u_m[boundary] + ny[boundary] * v_m[boundary] + nz[boundary] * w_m[boundary]

        # compute perfectly reflecting boundary conditions
        u_p[boundary] = u_m[boundary] - 2.0 * (ndotum) * nx[boundary]
        v_p[boundary] = v_m[boundary] - 2.0 * (ndotum) * ny[boundary]
        w_p[boundary] = w_m[boundary] - 2.0 * (ndotum) * nz[boundary]
        p_p[boundary] = p_m[boundary]

        # compute boundary pressure using impedance: p = Z * u_n
        #p_p[boundary] = self.surface_impedance * ndotum
        
        # compute reflected normal velocity using p_p = Z * u_n => solve for u_n
        # ndotup = p_p[boundary] / self.surface_impedance
        # in this case I can just set u_n+ = u_n-
        #ndotup = ndotum
        
        # convert normal velocity to components
        #u_p[boundary] = u_m[boundary] + (ndotup - ndotum) * nx[boundary]
        #v_p[boundary] = v_m[boundary] + (ndotup - ndotum) * ny[boundary]
        #w_p[boundary] = w_m[boundary] + (ndotup - ndotum) * nz[boundary]

        # apply the source boundary term if source is still on
        if time < self.source_duration:
            p_p = self._apply_source_boundary_condition(time, p_p)
           
        # reshape for matrix-matrix multiplication
        u_m, v_m, w_m, p_m = self._reshape_to_rectangular(u_m, v_m, w_m, p_m)
        u_p, v_p, w_p, p_p = self._reshape_to_rectangular(u_p, v_p, w_p, p_p)
        return u_p, v_p, w_p, p_p, u_m, v_m, w_m, p_m 

    def _compute_homogeneous_material_flux(self):
        # homogeneous material fluxes
        flux_p = 0.5 * ((ndotup - ndotum) - (p_p - p_m))
        flux_u = 0.5 * (self.mesh.nx * ((p_p - p_m) - (ndotup - ndotum)))
        flux_v = 0.5 * (self.mesh.ny * ((p_p - p_m) - (ndotup - ndotum)))
        flux_w = 0.5 * (self.mesh.nz * ((p_p - p_m) - (ndotup - ndotum)))

    def _compute_upwind_flux(self):
        # RH-condition flux from Wenzhong Cao 2024 - Acoustic wave simulation in strongly heterogeneous...
        Z_p = rho_p * c_p
        Z_m = rho_m * c_m
        normal_vel_jump = ndotup - ndotum
        pressure_jump = p_p - p_m

        numerator = pressure_jump - Z_p * normal_vel_jump
        denominator = Z_p + Z_m
        correction = numerator / denominator

        flux_p = Z_m * (-ndotum - correction)
        flux_u = self.mesh.nx * (-(p_m / rho_m) + c_m * correction)
        flux_v = self.mesh.ny * (-(p_m / rho_m) + c_m * correction)
        flux_w = self.mesh.nz * (-(p_m / rho_m) + c_m * correction)

    def compute_rhs(self, u=None, v=None, w=None, p=None, time=0.0):
        """
        flux function based on Xijun He et al. 2025
        "An effective discontinuous Galerkin method for solving
        acoustic wave equations"
        page 6/14
        """
        # use stored fields in first RK time step iteration
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if p is None:
            p = self.p

        # get heterogeneous material matrices
        rho_p, rho_m, c_p, c_m = self._get_material_face_values()

        # spatial derivative matrices
        Dr = self.mesh.reference_element_operators.r_differentiation_matrix
        Ds = self.mesh.reference_element_operators.s_differentiation_matrix
        Dt = self.mesh.reference_element_operators.t_differentiation_matrix

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
        u_p, v_p, w_p, p_p, u_m, v_m, w_m, p_m = self._apply_boundary_conditions(time)

        # compute normal velocity at interior boundary and exterior boundary 
        ndotum = self.mesh.nx * u_m + self.mesh.ny * v_m + self.mesh.nz * w_m
        ndotup = self.mesh.nx * u_p + self.mesh.ny * v_p + self.mesh.nz * w_p

        # get max speed for every interface
        mu = np.maximum(c_p, c_m)

        # compute fluxes
        # flux from Xiun He 2025 - An effective discontinuous galerkin
        flux_p = 0.5 * (
            (rho_p * c_p**2 * u_p - rho_m * c_m**2 * u_m) * self.mesh.nx + \
            (rho_p * c_p**2 * v_p - rho_m * c_m**2 * v_m) * self.mesh.ny + \
            (rho_p * c_p**2 * w_p - rho_m * c_m**2 * w_m) * self.mesh.nz - \
            mu * (p_p - p_m)
        )
        flux_u = 0.5 * (self.mesh.nx * (((p_p / rho_p) - (p_m / rho_m)) - mu * (ndotup - ndotum)))
        flux_v = 0.5 * (self.mesh.ny * (((p_p / rho_p) - (p_m / rho_m)) - mu * (ndotup - ndotum)))
        flux_w = 0.5 * (self.mesh.nz * (((p_p / rho_p) - (p_m / rho_m)) - mu * (ndotup - ndotum)))

        ## get necessary matricies for integral computation
        face_scale = self.mesh.surface_to_volume_jacobian
        lift = self.mesh.reference_element_operators.lift_matrix

        ## inverse density and bulk modulus (rho c^2)
        inv_rho = 1.0 / self.mesh.density
        bulk = self.mesh.density * (self.mesh.speed ** 2)

        self.rhs_u = -inv_rho * dpdx - lift @ (face_scale * flux_u)
        self.rhs_v = -inv_rho * dpdy - lift @ (face_scale * flux_v)
        self.rhs_w = -inv_rho * dpdz - lift @ (face_scale * flux_w)
        self.rhs_p = -bulk * (dudx + dvdy + dwdz) - lift @ (face_scale * flux_p)

        return self.rhs_u, self.rhs_v, self.rhs_w, self.rhs_p,
