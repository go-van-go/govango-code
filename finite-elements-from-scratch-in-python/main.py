from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizing import *

# create finite element
dimension = 3
polynomial_order = 10
lagrange_element = LagrangeElement(dimension, polynomial_order)


# select mesh
#mesh_file = "./inputs/meshes/simple.msh"
mesh_file = "./inputs/meshes/0.2.msh"
#mesh_file = "./inputs/meshes/0.1.msh"
#mesh_file = "./inputs/meshes/0.05.msh"
#mesh_file = "./inputs/meshes/ultra_fine.msh"
#mesh_file = "./inputs/meshes/positive_default.msh"
#mesh_file = "./inputs/meshes/negative_default.msh"
#mesh_file = "./inputs/meshes/structured_cube.msh"

# create mesh
mesh = Mesh3d(mesh_file, lagrange_element)

# select physics
physics = LinearAcoustics(mesh)

# select time stepping method
t_initial = 0
t_final = 0.01
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

# when and how to visualize
save = True
interactive = True
visualize_start = 0
skips_between_interactive_visualization = 10
skips_between_saves = 1

# what to visualize
elements=[]
normals = elements
boundary_nodes=False
boundary_normals=False
boundary_face_nodes=False
mesh_edges=False
mesh_boundary=False
    

while time_stepper.t < time_stepper.t_final:
    # solution visualization (changes every time step)
    solution = time_stepper.physics.p # + physics.v + physics.w
    #solution = np.array([])
    #average_solution=time_stepper.physics.u
    average_solution= np.array([])
    jumps=np.array([])
    boundary_jumps=np.array([])

    if time_stepper.current_time_step >= visualize_start:
        if time_stepper.current_time_step % skips_between_saves == 0 and save: 
            visualize_mesh(mesh,
                           file_name=f"solution_t_{time_stepper.current_time_step:0>8}.png",
                           jumps=jumps,
                           normals=normals,
                           solution=solution,
                           average_solution=average_solution,
                           elements=elements,
                           boundary_nodes=boundary_nodes,
                           boundary_face_nodes=boundary_face_nodes,
                           boundary_normals=boundary_normals,
                           boundary_jumps=boundary_jumps,
                           mesh_edges=mesh_edges,
                           mesh_boundary=mesh_boundary,
                           save=True)

        if time_stepper.current_time_step % skips_between_interactive_visualization == 0 and interactive:
            visualize_mesh(mesh,
                           jumps=jumps,
                           normals=normals,
                           solution=solution,
                           average_solution=average_solution,
                           elements=elements,
                           boundary_nodes=boundary_nodes,
                           boundary_face_nodes=boundary_face_nodes,
                           boundary_normals=boundary_normals,
                           boundary_jumps=boundary_jumps,
                           mesh_edges=mesh_edges,
                           mesh_boundary=mesh_boundary,
                           save=False)

    time_stepper.advance_time_step()

    print(f"max solution: {np.max(solution)},   min:{np.min(solution)}")
    print(f"max dp: {np.max(time_stepper.physics.dp)},   min:{np.min(time_stepper.physics.dp)}")


    # Save the self instance to a file
    #with open(f'./outputs/sim_data_{self.current_time_step:0>8}.pkl', 'wb') as file:
    #    pickle.dump(self, file)

