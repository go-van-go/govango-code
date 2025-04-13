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
        if load_file:
            with open(load_file, 'rb') as file:
                self.time_stepper = pickle.load(file)
        elif time_stepper:
            self.time_stepper = time_stepper
        else:
            raise ValueError("Must provide either a `time_stepper` or a `load_file`.")
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

    def set_save_intervals(self, image=None, data=None, points=None, vtk=None):
        if image is not None:
            self.save_image_interval = image
        if data is not None:
            self.save_data_interval = data
        if points is not None:
            self.save_points_interval = points
        if vtk is not None:
            self.save_vtk_interval = vtk

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
            if self.save_vtk_interval and t_step % self.save_vtk_interval == 0:
               self._save_to_vtk(self.physics.p, resolution=40)

            #self.time_stepper.advance_time_step_rk_with_force_term()
            self.time_stepper.advance_time_step()
            self._log_info()

    def _save_data(self):
        visualizer = self.visualizer  # backup
        self.visualizer = None        # remove for pickling
        file_name=f't_{self.time_stepper.current_time_step:0>8}'
        with open(f'./outputs/data/{file_name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        self.visualizer = visualizer  # restore after saving


    def _save_image(self):
        self.visualizer.plotter.clear()
        self.visualizer._show_grid()
        #self.visualizer.add_inclusion_boundary()
        self.visualizer.add_cell_averages(self.time_stepper.physics.p)
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

    def _log_info(self):
        sys.stdout.write(f"\rTimestep: {self.time_stepper.current_time_step}, Time: {self.time_stepper.t:.6f}")
        sys.stdout.flush()
