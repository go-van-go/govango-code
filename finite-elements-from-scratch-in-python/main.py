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

# create finite element
dimension = 3
polynomial_order = 2 # eta1 and eta2 are defined for N=2
lagrange_element = LagrangeElement(dimension, polynomial_order)

# select mesh
#mesh_file = "./inputs/meshes/simple.msh"
#mesh_file = "./inputs/meshes/0.1.msh"
#mesh_file = "./inputs/meshes/0.05.msh"
#mesh_file = "./inputs/meshes/split03.msh"
#mesh_file = "./inputs/meshes/split05.msh"
#mesh_file = "./inputs/meshes/split1.msh"
mesh_file = "./inputs/meshes/015_split_source.msh"
#mesh_file = "./inputs/meshes/long_source.msh"


# create mesh
mesh = Mesh3d(mesh_file, lagrange_element)

# select physics
physics = LinearAcoustics(mesh)

# select time stepping method
t_initial = 0.0
t_final = 2.0
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

# when and how to visualize
save_image = True
view_interactive = False
save_data = True 
visualize_start = 0
skips_between_interactive_visualization = 100
skips_between_data_saves= 100
skips_between_image_saves = 5

def load_file(file_path):
    """ load a timestepper from memory """
    with open(file_path, 'rb') as file:
        time_stepper = pickle.load(file)
    return time_stepper

file_path = f'./outputs/3d_data/time_stepper_00001700.pkl'
time_stepper = load_file(file_path)

pre_visualizer = Visualizer(time_stepper, save=False)
pre_visualizer.add_field_3d(time_stepper.physics.p, resolution=40)
pre_visualizer.show()
pre_visualizer.plotter.clear()

saved_visualizer = Visualizer(time_stepper, save=True)
interactive_visualizer = Visualizer(time_stepper, save=False)
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
            saved_visualizer.save()
            saved_visualizer.plotter.clear()
        # visualize interactive plot
        if time_stepper.current_time_step % skips_between_interactive_visualization == 0 and view_interactive:
            saved_visualizer._show_grid()
            interactive_visualizer = Visualizer(time_stepper, save=False)
            #interactive_view_interactive.add_cell_averages(physics.p)
            interactive_visualizer.add_nodes_3d(physics.p)
            interactive_visualizer.show()
            interactive_visualizer.plotter.clear()

    # save time_stepper object
    if time_stepper.current_time_step % skips_between_data_saves == 0 and save_data:
        # Save the self instance to a file
        with open(f'./outputs/3d_data/time_stepper_{time_stepper.current_time_step:0>8}.pkl', 'wb') as file:
            pickle.dump(time_stepper, file)

    time_stepper.advance_time_step_rk_with_force_term()
    #time_stepper.advance_time_step()

    # print timestep 
    sys.stdout.write(f"\rTimestep: {time_stepper.current_time_step}, Time: {time_stepper.t:.6f}")
    sys.stdout.flush()

gmsh.finalize()
