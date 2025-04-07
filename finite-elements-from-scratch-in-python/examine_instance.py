import pickle
import sys
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer

# Create finite element
dimension = 3
polynomial_order = 2 # eta1 and eta2 are defined for N=2
lagrange_element = LagrangeElement(dimension, polynomial_order)

# Select mesh file
#mesh_file = "./inputs/meshes/simple.msh"
#mesh_file = "./inputs/meshes/0.1.msh"
#mesh_file = "./inputs/meshes/0.05.msh"
#mesh_file = "./inputs/meshes/split03.msh"
#mesh_file = "./inputs/meshes/split05.msh"
#mesh_file = "./inputs/meshes/split1.msh"
#mesh_file = "./inputs/meshes/015_split_source.msh"
#mesh_file = "./inputs/meshes/0.05_split_source.msh"
mesh_file = "./inputs/meshes/0.05_small_split_source.msh"
#mesh_file = "./inputs/meshes/0.05_heterogeneous_source.msh"
#mesh_file = "./inputs/meshes/long_source.msh"

# Create mesh
mesh = Mesh3d(mesh_file, lagrange_element)

# Create physics
physics = LinearAcoustics(mesh)

# Create time stepping object 
t_initial = 0.0
t_final = 1.6

file_path = f'./outputs/3d_data/time_stepper_00002000.pkl'
#file_path = False

def load_file(file_path):
    """ load a timestepper from memory """
    with open(file_path, 'rb') as file:
        time_stepper = pickle.load(file)
    return time_stepper

if file_path:
    time_stepper = load_file(file_path)
else:
    time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)
    

# Save and visualization parameters 
save_image = True
view_interactive = True 
save_data = True
visualize_start = 0
skips_between_interactive_visualization = 100
skips_between_data_saves= 100
skips_between_image_saves = 10

# Pre Visualization
pre_visualizer = Visualizer(time_stepper, save=False)
#pre_visualizer.add_field_3d(time_stepper.physics.p, resolution=50)
#pre_visualizer.add_cell_averages(time_stepper.physics.p)
pre_visualizer.add_nodes_3d(time_stepper.physics.p)
#pre_visualizer.add_wave_speed()
pre_visualizer.show()
#pre_visualizer.plotter.clear()

# Create visualizers
saved_visualizer = Visualizer(time_stepper, save=True)
interactive_visualizer = Visualizer(time_stepper, save=False)

# Keep track of field values at points 
points = [[0.5, 0.5, 0.10],
          [0.5, 0.5, 0.30]]
fast_data = np.zeros((720))
slow_data = np.zeros((720))
counter = 0

breakpoint()

# Main time loop of simulation
while time_stepper.t < time_stepper.t_final:
    # only visualize after visualize_start
    if time_stepper.current_time_step >= visualize_start:
        # visualize plot to be saved
        if time_stepper.current_time_step % skips_between_image_saves == 0 and save_image: 
            saved_visualizer._show_grid()
            saved_visualizer.add_cell_averages(physics.p)
            #saved_visualizer.add_nodes_3d(physics.p)
            #saved_visualizer.add_nodes_3d(physics.p)
            saved_visualizer.save()
            saved_visualizer.plotter.clear()

            slow_data[counter] = saved_visualizer.eval_at_point(points[0][0], points[0][1], points[0][2], time_stepper.physics.p)
            fast_data[counter] = saved_visualizer.eval_at_point(points[1][0], points[1][1], points[1][2], time_stepper.physics.p)
            counter += 1

        # visualize interactive plot
        if time_stepper.current_time_step % skips_between_interactive_visualization == 0 and view_interactive:
            with open(f'./outputs/3d_data/time_stepper_{time_stepper.current_time_step:0>8}.pkl', 'wb') as file:
                pickle.dump(time_stepper, file)

            #interactive_visualizer = Visualizer(time_stepper, save=False)
            ##interactive_view_interactive.add_cell_averages(physics.p)
            #interactive_visualizer.add_nodes_3d(physics.p)
            #interactive_visualizer.show()
            #interactive_visualizer.plotter.clear()

    if time_stepper.current_time_step % skips_between_data_saves == 0 and save_data:
        # Save the self instance to a file
        #with open(f'./outputs/3d_data/time_stepper_{time_stepper.current_time_step:0>8}.pkl', 'wb') as file:
        #    pickle.dump(time_stepper, file)
        np.save('slow_data.npy', slow_data)
        np.save('fast_data.npy', fast_data)

    time_stepper.advance_time_step_rk_with_force_term()
    #time_stepper.advance_time_step()

    # print timestep 
    sys.stdout.write(f"\rTimestep: {time_stepper.current_time_step}, Time: {time_stepper.t:.6f}")
    sys.stdout.flush()

gmsh.finalize()
