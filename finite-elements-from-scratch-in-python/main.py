import pickle
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
#mesh_file = "./inputs/meshes/structured_cube.msh"
#mesh_file = "./inputs/meshes/10.msh"
#mesh_file = "./inputs/meshes/0.2.msh"
mesh_file = "./inputs/meshes/0.1.msh"
#mesh_file = "./inputs/meshes/0.06.msh"
#mesh_file = "./inputs/meshes/0.05.msh"
#mesh_file = "./inputs/meshes/0.025.msh"
#mesh_file = "./inputs/meshes/split03.msh"
#mesh_file = "./inputs/meshes/split05.msh"
#mesh_file = "./inputs/meshes/split1.msh"

# create mesh
mesh = Mesh3d(mesh_file, lagrange_element)
# select physics
physics = LinearAcoustics(mesh)

# select time stepping method
t_initial = 0
t_final = 2
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

# when and how to visualize
save = True
interactive = True 
interactive_save = True 
visualize_start = 0
skips_between_interactive_visualization = 20
skips_between_interactive_saves= 100
skips_between_saves = 5

## Specify the path to the saved pickle file
#file_path = f'./outputs/3d_data/sim_data_00000500.pkl'
#
## Open the file in read-binary mode and load the object
#with open(file_path, 'rb') as file:
#    physics = pickle.load(file)
#
#time_stepper = LowStorageRungeKutta(physics, 0.1345, t_final)
#time_stepper.current_time_step = 700

visualizer = Visualizer(time_stepper)
#visualizer.add_field_3d(physics.p, 10)
#visualizer.add_field_point_cloud(physics.p, 20)
#visualizer.add_nodes_3d(physics.p)
#visualizer.add_cells([4,5,6])
#visualizer.add_cell_averages(physics.p)
#visualizer.add_mesh()
#visualizer.add_mesh_boundary()
#visualizer.add_inclusion_boundary()
#visualizer.show()

while time_stepper.t < time_stepper.t_final:

    if time_stepper.current_time_step >= visualize_start:
        if time_stepper.current_time_step % skips_between_saves == 0 and save: 
            breakpoint()
            visualizer.add_cell_averages(physics.p)
            visualizer.save()
        if time_stepper.current_time_step % skips_between_interactive_visualization == 0 and interactive:
            pass
    if time_stepper.current_time_step % skips_between_interactive_saves == 0 and interactive_save:
        # Save the self instance to a file
        with open(f'./outputs/3d_data/sim_data_{time_stepper.current_time_step:0>8}.pkl', 'wb') as file:
            pickle.dump(time_stepper.physics, file)

    #time_stepper.advance_time_step_rk_with_force_term()
    time_stepper.advance_time_step()

    print(f"t = {time_stepper.t},  timestep = {time_stepper.current_time_step}")

    #print(f"max solution: {np.max(solution)},   min:{np.min(solution)}")
    #print(f"max dp: {np.max(time_stepper.physics.dp)},   min:{np.min(time_stepper.physics.dp)}")

gmsh.finalize()
