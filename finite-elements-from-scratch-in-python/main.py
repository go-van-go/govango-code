from wave_simulator.reference_elements import LagrangeElement
from wave_simulator.reference_element_operators import ReferenceElementOperators
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_stepper import LowStorageRungaKutta
from wave_simulator.simulator import Simulator

# create finite element
dimension = 3
polynomial_order = 3
lagrange_element = LagrangeElement(dimension, polynomial_order)

# create Mesh
mesh_file = "./inputs/simple.msh"
mesh = Mesh3d(mesh_file, LagrangeElement)

# select physics
physics = LinearAcoustics()

# select time stepping method
t_initial = 0
t_final = 10
time_stepper = LowStorageRungaKutta(physics, t_initial, t_final)

# create simulator
simulation = Simulator(mesh, physics, time_stepper)

# run the simulation
simulation.run()


