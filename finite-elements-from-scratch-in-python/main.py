from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizing import *

# create finite element
dimension = 3
polynomial_order = 3
lagrange_element = LagrangeElement(dimension, polynomial_order)

# create Mesh
mesh_file = "./inputs/meshes/fine_grid.msh"
mesh = Mesh3d(mesh_file, lagrange_element)

# select physics
physics = LinearAcoustics(mesh)

# select time stepping method
t_initial = 0
t_final = 1
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

while time_stepper.t < time_stepper.t_final:
    time_stepper.advance_time_step()
    # Save the self instance to a file
    #with open(f'./outputs/sim_data_{self.current_time_step:0>8}.pkl', 'wb') as file:
    #    pickle.dump(self, file)
    visualize_mesh(time_stepper.physics.mesh, file_name=f"solution_t_{time_stepper.current_time_step:0>8}.png", solution=time_stepper.physics.p, save=True)

