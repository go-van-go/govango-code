import sys
import math
import pickle
import time
import numpy as np
from wave_simulator.mesh import Mesh3d
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer
from wave_simulator.spatial_evaluator import SpatialEvaluator

class Simulator:
    def __init__(self,
                 time_stepper: LowStorageRungeKutta,
                 output_path,
                 save_image_interval,
                 save_points_interval,
                 save_data_interval,
                 save_energy_interval,
                 pressure_reciever_locations,
                 u_velocity_reciever_locations,
                 v_velocity_reciever_locations,
                 w_velocity_reciever_locations):
        self.output_path = output_path
        self.time_stepper = time_stepper
        self.physics = self.time_stepper.physics
        self.mesh = self.physics.mesh
        self.spatial_evaluator = SpatialEvaluator(self.mesh)
        self.t_final = self.time_stepper.t_final

        self.save_image_interval = save_image_interval
        self.save_data_interval = save_data_interval
        self.save_points_interval = save_points_interval
        self.save_energy_interval = save_energy_interval

        self.track_points(
            pressure_reciever_locations,
            u_velocity_reciever_locations,
            v_velocity_reciever_locations,
            w_velocity_reciever_locations
        )

        self.energy_index = 0
        self._get_source_data()

        self.data = self._get_data()
        self.visualizer = Visualizer(self.data)
        self.initialize_energy_array()

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
        start_time = time.time()
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

            #self.time_stepper.advance_time_step_rk_with_force_term()
            self.time_stepper.advance_time_step()
            self._log_info(start_time)

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

    def _get_data(self):
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
            'initialize_gmsh': self.mesh.initialize_gmsh,
            'speed_per_cell': self.mesh.speed[0,:],  # First row only
            'density_per_cell': self.mesh.density[0,:]  # First row only
        }

        # Include simulator tracking data if available
        simulator_data = {}
        if hasattr(self, 'tracked_fields') and self.tracked_fields:
            simulator_data['tracked_fields'] = self.tracked_fields
        if hasattr(self, 'energy_data') and self.energy_data is not None:
            simulator_data['energy_data'] = self.energy_data
        if hasattr(self, 'kinetic_data') and self.kinetic_data is not None:
            simulator_data['kinetic_data'] = self.kinetic_data
        if hasattr(self, 'potential_data') and self.potential_data is not None:
            simulator_data['potential_data'] = self.potential_data
        if hasattr(self, 'source_data') and self.source_data is not None:
            simulator_data['source_data'] = self.source_data

        data = {
            'current_time_step': self.time_stepper.current_time_step,
            'current_time': self.time_stepper.t,
            'dt': self.time_stepper.dt,
            't_final': self.t_final,
            'fields': {
                'p': self.physics.p,
                'u': self.physics.u,
                'v': self.physics.v,
                'w': self.physics.w
            },
            'mesh': mesh_data,
            'simulator': simulator_data,
            'output_path': self.output_path,
            'save_image_interval' : self.save_image_interval,
            'save_data_interval' : self.save_data_interval,
            'save_points_interval': self.save_points_interval,
            'save_energy_interval': self.save_energy_interval,
        }
        return data
        
    def _save_data(self):
        if not hasattr(self, '_save_index'):
            self._save_index = 0  # initialize the save index
    
        self.data = self._get_data()
        timestep_str = f'{self.time_stepper.current_time_step:0>8}'
        save_index_str = f'{self._save_index:0>8}'
    
        file_name = f'{save_index_str}_t{timestep_str}.pkl'
        file_path = f'{self.output_path}/data/{file_name}'
    
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        self._save_index += 1  # increment the save index after saving

    def _save_image(self):
        self.data = self._get_data()
        self.visualizer.set_data(self.data)
        self.visualizer.save()

    def _save_tracked_points(self):
        for name, field in self.tracked_fields.items():
            values = field["data"]
            points = field["points"]
            field_array = getattr(self.physics, field["field_name"])
    
            for i, (x, y, z) in enumerate(points):
                values[i, self.column_index] = self.spatial_evaluator.eval_at_point(x, y, z, field_array)
    
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

    def _log_info(self, start_time):
        runtime = time.time() - start_time
        sys.stdout.write(f"\rTimestep: {self.time_stepper.current_time_step}, Time: {self.time_stepper.t:.6f}, Runtime: {runtime:.2f}s")
        sys.stdout.flush()
