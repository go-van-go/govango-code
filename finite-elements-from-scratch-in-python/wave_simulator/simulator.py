import sys
import math
import pickle
import numpy as np
from wave_simulator.mesh import Mesh3d
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer

class Simulator:
    def __init__(self, time_stepper: LowStorageRungeKutta = None, load_file: str = None):
        self.time_stepper = time_stepper
        self.physics = self.time_stepper.physics
        self.mesh = self.physics.mesh
        self.t_final = self.time_stepper.t_final
        self.visualizer = Visualizer(self.time_stepper)

        self.save_image_interval = 0
        self.visualizer.save_image_interval = 0
        self.save_data_interval = 0
        self.save_points_interval = 0
        self.save_vtk_interval = 0

        self.tracked_points = []
        self.energy_index = 0
        self._get_source_data()

    def set_save_intervals(self,
                           image=None,
                           data=None,
                           points=None,
                           energy=None,
                           vtk=None):
        if image is not None:
            self.save_image_interval = image
        if data is not None:
            self.save_data_interval = data
        if points is not None:
            self.save_points_interval = points
        if energy is not None:
            self.save_energy_interval = energy
            self.initialize_energy_array()
        if vtk is not None:
            self.save_vtk_interval = vtk

    def initialize_energy_array(self):
        self.num_readings = math.ceil(self.time_stepper.num_time_steps / self.save_energy_interval)
        self.energy_data = np.zeros(self.num_readings)
        self.kinetic_data = np.zeros(self.num_readings)
        self.potential_data = np.zeros(self.num_readings)

    def track_points(self, points):
        # save points to be tracked
        self.tracked_points = points
        # initialize array to hold all values
        self.num_readings = math.ceil(self.time_stepper.num_time_steps / self.save_points_interval)
        self.point_data = np.zeros((len(points), self.num_readings))
        # initialize column index for indexing
        self.column_index = 0

    def run(self):
        while self.time_stepper.t < self.t_final:
            t_step = self.time_stepper.current_time_step

            # check save intervalues and save accordingly  
            if self.save_image_interval and t_step % self.save_image_interval == 0:
                self._save_image()
            if self.save_data_interval and t_step % self.save_data_interval == 0:
                self._save_data()
            if self.save_points_interval and t_step % self.save_points_interval == 0:
                self._save_tracked_points()
            if self.save_energy_interval and t_step % self.save_energy_interval == 0:
                self._save_energy()
            if self.save_vtk_interval and t_step % self.save_vtk_interval == 0:
               self._save_to_vtk(self.physics.p, resolution=40)

            #self.time_stepper.advance_time_step_rk_with_force_term()
            self.time_stepper.advance_time_step()
            self._log_info()

    def _get_source_data(self):
        num_time_steps = self.time_stepper.num_time_steps
        self.source_data = np.zeros(num_time_steps)
        for time in range(num_time_steps):
            t = time*self.time_stepper.dt
            self.source_data[time] = self.physics._get_amplitude(t)

    def _save_data(self):
        visualizer = self.visualizer  # backup
        self.visualizer = None        # remove for pickling
        file_name=f't_{self.time_stepper.current_time_step:0>8}'
        with open(f'./outputs/data/{file_name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        self.visualizer = visualizer  # restore after saving

    def _save_image(self):
        if self.visualizer == None:
            self.visualizer = Visualizer(self.time_stepper)
        self.visualizer.plotter.clear()
        self.visualizer._show_grid()
        self.visualizer.add_inclusion_boundary()
        #self.visualizer.add_cell_averages(self.time_stepper.physics.p)
        self.visualizer.add_nodes_3d(self.time_stepper.physics.p)
        self.visualizer.save()

    def _save_to_vtk(self, field, resolution=40):
        self.visualizer.save_to_vtk(field, resolution)

    def _save_tracked_points(self):
        # get field
        field = self.physics.p
        # Sample field at tracked points
        for i, point in enumerate(self.tracked_points):
            value = self.visualizer.eval_at_point(point[0], point[1], point[2], field)
            self.point_data[i,self.column_index] = value
        # increment column index
        self.column_index += 1

    def _save_energy(self):
        # on page 37 in Hesthaven and warburton we see how the mass matrix can be
        # used to recover the energy (l2 norm) of the system
        # uT M u = || u ||^2
        # get nodal values
        j = self.mesh.jacobians[0,:]  # shape (K,)
        p = self.physics.p
        u = self.physics.u
        v = self.physics.v
        w = self.physics.w
        mass = self.mesh.reference_element_operators.mass_matrix
        num_cells = self.mesh.num_cells
        rho = self.mesh.density[0,:]  # shape (Np, K)
        c = self.mesh.speed[0,:]      # shape (Np, K)
        inv_bulk = 1.0 / (rho * (c**2))  # shape (Np, K)
        
        # potential energy: (1/2) * p^2 / (rho * c^2)
        potential = np.array([p[:, i].T @ mass @ p[:, i] for i in range(num_cells)])
        potential = 0.5 * inv_bulk * j * potential
        self.potential_data[self.energy_index] = np.sum(potential)
 
        # kinetic energy: (1/2) * rho * (u^2 + v^2 + w^2)
        kinetic_u = np.array([u[:, i].T @ mass @ u[:, i] for i in range(num_cells)])
        kinetic_v = np.array([v[:, i].T @ mass @ v[:, i] for i in range(num_cells)])
        kinetic_w = np.array([w[:, i].T @ mass @ w[:, i] for i in range(num_cells)])
        kinetic = (0.5 * rho * j * (kinetic_u + kinetic_v + kinetic_w))
        self.kinetic_data[self.energy_index] = np.sum(kinetic)
    
        # total energy
        energy = np.sum(potential + kinetic)
        self.energy_data[self.energy_index] = energy

        self.energy_index += 1
        

    def _log_info(self):
        sys.stdout.write(f"\rTimestep: {self.time_stepper.current_time_step}, Time: {self.time_stepper.t:.6f}")
        sys.stdout.flush()
