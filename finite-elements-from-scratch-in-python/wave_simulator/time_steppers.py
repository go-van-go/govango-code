from wave_simulator.Physics import LinearAcoustics


class LowStorageRungeKutta:
    def __init__(self, physics: LinearAcoustics, t_initial, t_final):
        self.t_initial = 0
        self.t_final = 10
        self.t = t_initial

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

        # Runge-Kutta residual storage
        self.res_u = np.zeros((Np, K))
        self.res_v = np.zeros((Np, K))
        self.res_w = np.zeros((Np, K))
        self.res_p = np.zeros((Np, K))

        self._compute_time_step_size()

    def _compute_time_step_size(self, N, face_scale):
        dt = 1.0 / (np.max(np.max(face_scale)) * N * N)
        # correct dt for integer # of time steps
        num_time_steps = int(np.ceil(self.t_final/ dt))
        self.dt = FinalTime / num_time_steps

    def advance_time_step(self):
        for i in range(0, 5):  # inner multi-stage Runge-Kutta loop
            # compute right hand side of TM-mode Maxwell's equations
            rhs_u = self.physics.rhs_u
            rhs_v = self.physics.rhs_v
            rhs_w = self.physics.rhs_w
            rhs_p = self.physics.rhs_p

            # initiate, increment Runge-Kutta residuals and update fields
            res_u = rk4a[i] * res_u + dt * rhs_u
            u = u + rk4b[i] * res_u
            res_v = rk4a[i] * res_v + dt * rhs_u
            v = v + rk4b[i] * res_v
            res_w = rk4a[i] * res_w + dt * rhs_w
            w = w + rk4b[i] * res_w
            res_p = rk4a[i] * res_p + dt * rhs_p
            p = p + rk4b[i] * res_p

        # Sum the magnitudes of H and E
        time += dt  # Increment time


if __name__ == "__main__":
    pass

