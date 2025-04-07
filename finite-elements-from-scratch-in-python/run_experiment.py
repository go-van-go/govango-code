from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.mesh import Mesh3d 
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.visualizer import Visualizer

# Select mesh file
#mesh_file = "./inputs/meshes/015_split_source.msh"
#mesh_file = "./inputs/meshes/0.05_split_source.msh"
mesh_file = "./inputs/meshes/0.05_small_split_source.msh"
#mesh_file = "./inputs/meshes/0.05_heterogeneous_source.msh"
#mesh_file = "./inputs/meshes/long_source.msh"

# Create finite element
dimension = 3
polynomial_order = 2 # eta1 and eta2 are defined for N=2
lagrange_element = LagrangeElement(dimension, polynomial_order)
# Create mesh
mesh = Mesh3d(mesh_file, lagrange_element)
# Create physics
physics = LinearAcoustics(mesh)
# Create time stepping object 
t_initial = 0.0
t_final = 1.6
time_stepper = LowStorageRungeKutta(physics, t_initial, t_final)

# initialize simulator object
simulator = Simulator(time_stepper)
simulator.set_save_intervals(image=10, data=100, points=10, vtk=100)
simulator.track_points([
    [0.5, 0.5, 0.10],
    [0.5, 0.5, 0.30],
])

# run experiment
simulator.run()
