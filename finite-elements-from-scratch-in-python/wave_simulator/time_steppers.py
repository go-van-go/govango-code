import pickle
import numpy as np
from wave_simulator.physics import LinearAcoustics
from wave_simulator.visualizing import visualize_mesh 


class LowStorageRungeKutta:
    def __init__(self, physics: LinearAcoustics, t_initial, t_final):
        self.t_initial = 0
        self.t_final = 10
        self.t = t_initial
        self.current_time_step = 0
        self.physics = physics

        # Runge-Kutta residual storage
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

        nodes_per_cell = physics.mesh.reference_element.nodes_per_cell 
        num_cells = physics.mesh.num_cells
        self.res_u = np.zeros((nodes_per_cell, num_cells))
        self.res_v = np.zeros((nodes_per_cell, num_cells))
        self.res_w = np.zeros((nodes_per_cell, num_cells))
        self.res_p = np.zeros((nodes_per_cell, num_cells))

        self._compute_time_step_size()

    def _compute_time_step_size(self):
        n = self.physics.mesh.reference_element.n
        surface_to_volume_jacobian = self.physics.mesh.surface_to_volume_jacobian
        dt = 1.0 / (np.max(np.max(surface_to_volume_jacobian)) * n * n)
        # correct dt for integer # of time steps
        num_time_steps = int(np.ceil(self.t_final/ dt))
        print(f"time step size: {dt}")
        self.dt = (self.t_final / num_time_steps)

    def advance_time_step(self):
        u = self.physics.u
        v = self.physics.v
        w = self.physics.w
        p = self.physics.p
        dt = self.dt

        for i in range(5):  # inner multi-stage Runge-Kutta loop
            rhs_u, rhs_v, rhs_w, rhs_p = self.physics.compute_rhs()

            # initiate, increment Runge-Kutta residuals and update fields
            self.res_u = self.rk4a[i] * self.res_u + dt * rhs_u
            u = u + self.rk4b[i] * self.res_u
            self.res_v = self.rk4a[i] * self.res_v + dt * rhs_v
            v = v + self.rk4b[i] * self.res_v
            self.res_w = self.rk4a[i] * self.res_w + dt * rhs_w
            w = w + self.rk4b[i] * self.res_w
            self.res_p = self.rk4a[i] * self.res_p + dt * rhs_p
            p = p + self.rk4b[i] * self.res_p

        self.physics.u = u
        self.physics.v = v
        self.physics.w = w
        self.physics.p = p

        self.t += self.dt  # Increment time
        self.current_time_step += 1
        print(self.current_time_step)


if __name__ == "__main__":
    pass

