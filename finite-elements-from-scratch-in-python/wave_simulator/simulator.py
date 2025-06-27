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

    def track_points(self, pressure=None, x=None, y=None, z=None):
        self.num_readings = math.ceil(self.time_stepper.num_time_steps / self.save_points_interval)
        self.column_index = 0
        self.tracked_fields = {}
        if pressure:
            self.tracked_fields["pressure"] = {
                "points": pressure,
                "data": np.zeros((len(pressure), self.num_readings)),
                "field_name": "p"
            }
        if x:
            self.tracked_fields["x"] = {
                "points": x,
                "data": np.zeros((len(x), self.num_readings)),
                "field_name": "u"
            }
        if y:
            self.tracked_fields["y"] = {
                "points": y,
                "data": np.zeros((len(y), self.num_readings)),
                "field_name": "v"
            }
        if z:
            self.tracked_fields["z"] = {
                "points": z,
                "data": np.zeros((len(z), self.num_readings)),
                "field_name": "w"
            }

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
            self.source_data[time] = self.physics._get_source_pressure(t)

#    def _save_data(self):
#        visualizer = self.visualizer  # backup
#        self.visualizer = None        # remove for pickling
#        file_name=f't_{self.time_stepper.current_time_step:0>8}'
#        with open(f'{self.output_path}/data/{file_name}.pkl', 'wb') as f:
#            pickle.dump(self, f)
#        self.visualizer = visualizer  # restore after saving

    def _save_data(self):
        # Create minimal mesh data for visualization
        mesh_data = {
            'x': self.mesh.x,
            'y': self.mesh.y,
            'z': self.mesh.z,
            'vertex_coordinates': self.mesh.vertex_coordinates,
            'cell_to_vertices': self.mesh.cell_to_vertices,
            'nx': self.mesh.nx,
            'ny': self.mesh.ny,
            'nz': self.mesh.nz,
            'reference_element': self.mesh.reference_element,
            'speed_per_cell': self.mesh.speed[0,:],  # First row only
            'density_per_cell': self.mesh.density[0,:]  # First row only
        }

        # Include simulator tracking data if available
        simulator_data = {}
        if hasattr(self, 'tracked_fields') and self.tracked_fields:
            simulator_data['tracked_fields'] = self.tracked_fields
        if hasattr(self, 'energy_data') and self.energy_data is not None:
            simulator_data['energy_data'] = self.energy_data[:self.energy_index] if self.energy_index > 0 else []
        if hasattr(self, 'kinetic_data') and self.kinetic_data is not None:
            simulator_data['kinetic_data'] = self.kinetic_data[:self.energy_index] if self.energy_index > 0 else []
        if hasattr(self, 'potential_data') and self.potential_data is not None:
            simulator_data['potential_data'] = self.potential_data[:self.energy_index] if self.energy_index > 0 else []
        if hasattr(self, 'source_data') and self.source_data is not None:
            simulator_data['source_data'] = self.source_data

        data = {
            'time_step': self.time_stepper.current_time_step,
            'time': self.time_stepper.t,
            'dt': self.time_stepper.dt,
            'fields': {
                'p': self.physics.p,
                'u': self.physics.u,
                'v': self.physics.v,
                'w': self.physics.w
            },
            'mesh': mesh_data,
            'simulator': simulator_data
        }

        file_name=f't_{self.time_stepper.current_time_step:0>8}'
        with open(f'{self.output_path}/data/t_{file_name}.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_image(self):
        if self.visualizer == None:
            self.visualizer = Visualizer(self.time_stepper)
        self.visualizer.plotter.clear()
        self.visualizer._show_grid()
        self.visualizer.add_inclusion_boundary()
        #self.visualizer.add_cell_averages(self.time_stepper.physics.p)
        self.visualizer.add_nodes_3d(self.time_stepper.physics.p)
        #self.visualizer.add_nodes_3d(self.time_stepper.physics.w)
        self.visualizer.save()

    def _save_to_vtk(self, field, resolution=40):
        self.visualizer.save_to_vtk(field, resolution)

    #def _save_tracked_points(self):
    #    # get field
    #    field = self.physics.p
    #    # Sample field at tracked points
    #    for i, point in enumerate(self.tracked_points):
    #        value = self.visualizer.eval_at_point(point[0], point[1], point[2], field[i])
    #        self.point_data[i,self.column_index] = value
    #    # increment column index
    #    self.column_index += 1

    def _save_tracked_points(self):
        for name, field in self.tracked_fields.items():
            values = field["data"]
            points = field["points"]
            field_array = getattr(self.physics, field["field_name"])
    
            for i, (x, y, z) in enumerate(points):
                values[i, self.column_index] = self.visualizer.eval_at_point(x, y, z, field_array)
    
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
