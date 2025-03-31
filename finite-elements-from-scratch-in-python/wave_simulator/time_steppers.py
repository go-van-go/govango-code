import pickle
import numpy as np
from wave_simulator.physics import LinearAcoustics
from wave_simulator.visualizing import visualize_mesh 


class LowStorageRungeKutta:
    def __init__(self, physics: LinearAcoustics, t_initial, t_final):
        self.t_initial = t_initial
        self.t_final = t_final
        self.t = t_initial
        self.current_time_step = 0
        # Runge-Kutta residual storage
        nodes_per_cell = physics.mesh.reference_element.nodes_per_cell 
        num_cells = physics.mesh.num_cells
        self.rk4a = np.array([0.0,
                              -567301805773.0 / 1357537059087.0,
                              -2404267990393.0 / 2016746695238.0,
                              -3550918686646.0 / 2091501179385.0,
                              -1275806237668.0 / 842570457699.0])

        self.rk4b = np.array([1432997174477.0 / 9575080441755.0,
                              5161836677717.0 / 13612068292357.0,
                              1720146321549.0 / 2090206949498.0,
                              3134564353537.0 / 4481467310338.0,
                              2277821191437.0 / 14882151754819.0])
        self.res_u = np.zeros((nodes_per_cell, num_cells))
        self.res_v = np.zeros((nodes_per_cell, num_cells))
        self.res_w = np.zeros((nodes_per_cell, num_cells))
        self.res_p = np.zeros((nodes_per_cell, num_cells))
        self.physics = physics
        self.old_compute_time_step_size()

    def old_compute_time_step_size(self):
        n = self.physics.mesh.reference_element.n
        surface_to_volume_jacobian = self.physics.mesh.surface_to_volume_jacobian
        c = self.physics.max_speed
        dt = 1.0 / (np.max(np.max(surface_to_volume_jacobian)) * n * n * c) 
        # correct dt for integer # of time steps
        num_time_steps = int(np.ceil(self.t_final/ dt))
        print(f"time step size: {dt}")
        self.dt = (self.t_final / num_time_steps)

    def _compute_time_step_size(self):
        """
        from Xijun He et al. 2023 - Modeling 3D elastic Wave Propagation
        page 6/14
        """
        d = self.physics.mesh.smallest_diameter
        c = self.physics.max_speed
        alpha = 0.500 # max is 0.503 for eta1 = 0.33 and eta2 = 0.87
        dt = (alpha * d) / c
        # correct dt for integer # of time steps
        num_time_steps = int(np.ceil(self.t_final/ dt))
        self.dt = (self.t_final / num_time_steps)/8
        print(f"time step size: {self.dt}")
        print(f"max speed: {c}")
        print(f"smallest diameter: {d}")


    #def advance_time_step_rk_with_force_term(self):
    #    """
    #    advance forward in time by dt
    #    Algorithm by Xijun He et al. 2023
    #    "modeling 3D elastic waves propagation in Ti Media"
    #    page 5/14
    #    """
    #    dt = self.dt
    #    tau = (3 - np.sqrt(3))/6
    #    eta1 = 0.33
    #    eta2 = 0.87
    #    x0, y0, z0 = (0.5, 0.5, 0.60)
    #    x_values = np.where(np.abs(self.physics.mesh.x - x0) < 0.02)[1]
    #    y_values = np.where(np.abs(self.physics.mesh.y - y0) < 0.02)[1]
    #    z_values = np.where(np.abs(self.physics.mesh.z - z0) < 0.04)[1]
    #    xy_values = np.intersect1d(x_values, y_values)
    #    xyz_values = np.intersect1d(xy_values, z_values)
    #    source_cell = xyz_values[0]
    #    source_node = 2
    #    source_frequency = 1
    #    amplitude = 50

    #    # compute K
    #    LC_u, LC_v, LC_w, LC_p = self.physics.compute_rhs()
    #    LC2_u, LC2_v, LC2_w, LC2_p = self.physics.compute_rhs(LC_u, LC_v, LC_w, LC_p)
    #    LC3_u, LC3_v, LC3_w, LC3_p = self.physics.compute_rhs(LC2_u, LC2_v, LC2_w, LC2_p)
    #     
    #    K_u = LC_u + tau * self.dt * LC2_u + eta1 * (tau * self.dt)**2 * LC3_u
    #    K_v = LC_v + tau * self.dt * LC2_v + eta1 * (tau * self.dt)**2 * LC3_v
    #    K_w = LC_w + tau * self.dt * LC2_w + eta1 * (tau * self.dt)**2 * LC3_w
    #    K_p = LC_p + tau * self.dt * LC2_p + eta1 * (tau * self.dt)**2 * LC3_p

    #    # compute Kbar
    #    T_u = self.physics.u + (1 - 2 * tau) * self.dt * K_u
    #    T_v = self.physics.v + (1 - 2 * tau) * self.dt * K_v
    #    T_w = self.physics.w + (1 - 2 * tau) * self.dt * K_w
    #    T_p = self.physics.p + (1 - 2 * tau) * self.dt * K_p
    #    LT_u, LT_v, LT_w, LT_p = self.physics.compute_rhs(T_u, T_v, T_w, T_p)
    #    LT2_u, LT2_v, LT2_w, LT2_p = self.physics.compute_rhs(LT_u, LT_v, LT_w, LT_p)
    #    LT3_u, LT3_v, LT3_w, LT3_p = self.physics.compute_rhs(LT2_u, LT2_v, LT2_w, LT2_p)

    #    Kbar_u = LT_u + tau * self.dt * LT2_u + eta2 * (tau * self.dt)**2 * LT3_u
    #    Kbar_v = LT_v + tau * self.dt * LT2_v + eta2 * (tau * self.dt)**2 * LT3_v
    #    Kbar_w = LT_w + tau * self.dt * LT2_w + eta2 * (tau * self.dt)**2 * LT3_w
    #    Kbar_p = LT_p + tau * self.dt * LT2_p + eta2 * (tau * self.dt)**2 * LT3_p

    #    # apply force term
    #    #K_p[source_node, source_cell] = amplitude * np.sin(2 * np.pi * source_frequency * (self.t + tau * self.dt))
    #    #Kbar_p[source_node, source_cell] = amplitude * np.sin(2 * np.pi * source_frequency * (self.t + (1 - tau) * self.dt))

    #    # update fields
    #    self.physics.u = self.physics.u + (self.dt/2) * (K_u + Kbar_u)
    #    self.physics.v = self.physics.v + (self.dt/2) * (K_v + Kbar_v)
    #    self.physics.w = self.physics.w + (self.dt/2) * (K_w + Kbar_w)
    #    self.physics.p = self.physics.p + (self.dt/2) * (K_p + Kbar_p)
    #    
    #    self.t += self.dt  # Increment time
    #    self.current_time_step += 1


    def advance_time_step(self):
        """advance forward in time by dt"""
        dt = self.dt

        for i in range(5):  # inner multi-stage Runge-Kutta loop
            rhs_u, rhs_v, rhs_w, rhs_p = self.physics.compute_rhs()

            # initiate, increment Runge-Kutta residuals and update fields
            self.res_u = self.rk4a[i] * self.res_u + dt * rhs_u
            self.physics.u = self.physics.u + self.rk4b[i] * self.res_u
            self.res_v = self.rk4a[i] * self.res_v + dt * rhs_v
            self.physics.v = self.physics.v + self.rk4b[i] * self.res_v
            self.res_w = self.rk4a[i] * self.res_w + dt * rhs_w
            self.physics.w = self.physics.w + self.rk4b[i] * self.res_w
            self.res_p = self.rk4a[i] * self.res_p + dt * rhs_p
            self.physics.p = self.physics.p + self.rk4b[i] * self.res_p

        self.t += self.dt  # Increment time
        self.current_time_step += 1
        print(self.current_time_step)

