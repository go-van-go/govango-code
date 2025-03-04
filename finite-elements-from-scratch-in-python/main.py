from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizing import *

# create finite element
dimension = 3
polynomial_order = 6
lagrange_element = LagrangeElement(dimension, polynomial_order)

# create Mesh
mesh_file = "./inputs/meshes/fine_grid.msh"
#mesh_file = "./inputs/meshes/simple.msh"
mesh = Mesh3d(mesh_file, lagrange_element)

# select physics
physics = LinearAcoustics(mesh)

# select time stepping method
t_initial = 0
t_final = 0.01
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

while time_stepper.t < time_stepper.t_final:
    solution = time_stepper.physics.u
    #elements = [i for i in range(200)]
    #elements = mesh.cell_to_cells[5:50].flatten().tolist()
    elements = []
    visualize_mesh(mesh,
                   file_name=f"solution_t_{time_stepper.current_time_step:0>8}.png",
                   solution=solution,
                   elements=elements,
                   boundary_nodes=False,
                   save=True)

    time_stepper.advance_time_step()

    # Save the self instance to a file
    #with open(f'./outputs/sim_data_{self.current_time_step:0>8}.pkl', 'wb') as file:
    #    pickle.dump(self, file)

